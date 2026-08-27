"""
main.py
=====================================================================
【概要】
highway-env 上でSeed値を完全固定した状態でシミュレーションを実行し、
「TrajectoryTransformer（上位プランナー）」が過去の時系列観測から
行動を提案し、「RSSChecker（安全性検証レイヤー）」がその行動を
検証・必要ならOverrideする、という一連の流れを統合した第3段階版です。

【第3段階での変更点】
    - ハイパーパラメータを config.py に集約し、
      collect_demo.py / train_irl.py と完全に一致させた
      （SEQ_LENなどがズレて動かなくなる事故を防ぐため）
    - 学習済みモデルの重み（config.MODEL_SAVE_PATH）が存在すれば
      自動的に読み込んで推論するように変更した
        - 存在する場合 → train_irl.pyで学習した「賢い」方策で走行する
        - 存在しない場合 → 従来通りランダム初期化された重みで走行する
          （常にSLOWERを選び続けるなど、単純な挙動になりやすい）
      この2つを比較することで、「IRL学習によって方策がどう変化したか」を
      同じmain.pyのまま確認できるようになっている。

【アーキテクチャの全体像】

    観測履歴(deque)
    過去 SEQ_LEN ステップ分の観測を保持
    shape: [SEQ_LEN, feature_dim]
            │
            ▼ テンソル化してバッチ次元を追加
    shape: [1, SEQ_LEN, feature_dim]
            │
            ▼
    ┌─────────────────────────┐
    │  TrajectoryTransformer     │ ← 上位プランナー
    │  (学習済み重みがあれば読み込む)  │    過去の時系列から次の行動を提案する
    └───────────┬─────────────┘
                │ proposed_action ( + IRL報酬スコア：ログ用)
                ▼
    ┌─────────────────────────┐
    │   RSSChecker                │ ← 安全性の最終防衛ライン
    │  (rss_checker.py)           │    危険なら強制的にブレーキへ上書きする
    └───────────┬─────────────┘
                │ final_action
                ▼
    ┌─────────────────────────┐
    │   highway-env                │ ← シミュレーション環境
    └─────────────────────────┘
=====================================================================
"""

import os
from collections import deque

import numpy as np
import torch

import gymnasium as gym
import highway_env  # noqa: F401  # "highway-v0"などの環境をgymに登録するための副作用インポート

import config
from seed_utils import set_global_seed
from rss_checker import RSSChecker, RSSParameters, is_adjacent_lane_clear
from transformer_planner import TrajectoryTransformer, to_action_onehot, ACTION_NAMES, NUM_ACTIONS
from eval_utils import new_action_counter, merge_action_counters, format_action_distribution


# =====================================================================
# 定数定義
# 実体は config.py に集約されており、ここではその値を読み出しているだけ。
# main.py単体を読んでも意味が分かるよう、あえてローカル変数にも代入している。
# =====================================================================
SEED = config.SEED
NUM_EPISODES = config.NUM_EPISODES
MAX_STEPS_PER_EPISODE = config.MAX_STEPS_PER_EPISODE

SEQ_LEN = config.SEQ_LEN
D_MODEL = config.D_MODEL
N_HEAD = config.N_HEAD
NUM_LAYERS = config.NUM_LAYERS
DIM_FEEDFORWARD = config.DIM_FEEDFORWARD

LOG_INTERVAL = config.LOG_INTERVAL

# highway-envのDiscreteMetaActionにおける行動ID（可読性のため定数化）
ACTION_LANE_LEFT = 0
ACTION_IDLE = 1
ACTION_LANE_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4


def flatten_observation(obs: np.ndarray) -> np.ndarray:
    """
    highway-envのKinematics観測（shape=[vehicles_count, features]の2次元配列）を
    1次元ベクトルに平坦化するヘルパー関数。

    Parameters
    ----------
    obs : np.ndarray
        shape = [vehicles_count, features]
        例：[5, 5]（周辺車両5台 × (presence, x, y, vx, vy)の5特徴量）

    Returns
    -------
    np.ndarray
        shape = [vehicles_count * features]（例：25次元）のfloat32ベクトル
    """
    return obs.flatten().astype(np.float32)


def build_input_tensor(
    history: deque, seq_len: int, feature_dim: int
) -> torch.Tensor:
    """
    観測履歴（deque）からTransformerへの入力テンソルを構築する関数。

    エピソード開始直後は、まだ seq_len 分の履歴が溜まっていないため、
    「最も古い（＝現時点で唯一存在する最初の）観測を複製してパディングする」
    という単純な方法で不足分を埋める。

    Parameters
    ----------
    history : deque
        各要素が shape=[feature_dim] のnp.ndarrayであるdeque。
    seq_len : int
        Transformerに入力する時系列長
    feature_dim : int
        1ステップあたりの特徴量次元数

    Returns
    -------
    torch.Tensor
        shape = [1, seq_len, feature_dim]
        バッチサイズ1として、Transformerにそのまま入力できる形状のテンソル
    """
    history_list = list(history)

    if len(history_list) < seq_len:
        pad_count = seq_len - len(history_list)
        padding = [history_list[0].copy() for _ in range(pad_count)]
        history_list = padding + history_list

    stacked = np.stack(history_list, axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    return tensor


def propose_action_by_model(
    model: TrajectoryTransformer, history: deque, seq_len: int, feature_dim: int
):
    """
    TrajectoryTransformer（上位プランナー）を使い、過去の観測履歴から
    次に取るべき行動を提案する関数。

    Parameters
    ----------
    model : TrajectoryTransformer
        学習済み（または未学習）の方策モデル
    history : deque
        観測履歴
    seq_len : int
        時系列長
    feature_dim : int
        特徴量次元数

    Returns
    -------
    tuple(int, float)
        (提案する行動ID, その行動に対するIRL報酬スコア)
    """
    input_tensor = build_input_tensor(history, seq_len, feature_dim)

    model.eval()
    with torch.no_grad():
        action_logits, state_repr = model(input_tensor)
        if config.REWARD_BLEND_BETA > 0.0:
            all_rewards = model.compute_all_action_rewards(state_repr)
            combined_scores = action_logits + config.REWARD_BLEND_BETA * all_rewards
            action_tensor = model.select_action(combined_scores, deterministic=True)
        else:
            action_tensor = model.select_action(action_logits, deterministic=True)
        action_id = int(action_tensor.item())

        action_onehot = to_action_onehot(action_id, NUM_ACTIONS).unsqueeze(0)
        reward_score = model.compute_reward(state_repr, action_onehot)
        reward_value = float(reward_score.item())

    return action_id, reward_value


def get_front_vehicle_info(env: gym.Env):
    """
    現在の自車（ego vehicle）と同一車線上にいる前方車両との
    「車間距離」「自車速度」「前方車両速度」を取得するヘルパー関数。

    Parameters
    ----------
    env : gym.Env
        highway-envの環境インスタンス

    Returns
    -------
    tuple(float, float, float) or None
        (車間距離[m], 自車速度[m/s], 前方車両速度[m/s])
        同一車線上に前方車両が存在しない場合は None を返す。
    """
    road = env.unwrapped.road
    ego_vehicle = env.unwrapped.vehicle

    front_vehicle, _ = road.neighbour_vehicles(ego_vehicle, ego_vehicle.lane_index)

    if front_vehicle is None:
        return None

    distance = float(front_vehicle.position[0] - ego_vehicle.position[0])
    v_rear = float(ego_vehicle.speed)
    v_front = float(front_vehicle.speed)

    return distance, v_rear, v_front


def load_model_weights_if_available(model: TrajectoryTransformer) -> bool:
    """
    config.MODEL_SAVE_PATH に学習済みの重みファイルが存在すれば読み込む。

    Parameters
    ----------
    model : TrajectoryTransformer
        重みをロードする対象のモデル（in-placeで更新される）

    Returns
    -------
    bool
        True: 学習済み重みを読み込んだ場合 / False: 見つからず未学習のまま使う場合
    """
    if os.path.exists(config.MODEL_SAVE_PATH):
        # map_location="cpu" を指定することで、GPUで学習した重みでも
        # CPU環境で問題なく読み込めるようにしている
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[main] 学習済みモデルを読み込みました: {config.MODEL_SAVE_PATH}")
        return True
    else:
        print(
            f"[main] 学習済みモデルが見つかりません ({config.MODEL_SAVE_PATH})。 "
            f"ランダム初期化された重みのまま推論します。\n"
            f"       → collect_demo.py → train_irl.py の順に実行すると、"
            f"学習済みモデルが生成されます。"
        )
        return False


def run_simulation() -> None:
    """
    highway-env環境を構築し、Seedを固定した上で
    「TrajectoryTransformer（上位プランナー）+ RSSChecker（安全性検証）」の
    統合パイプラインを実行するメイン関数。
    """
    set_global_seed(SEED)

    env = gym.make("highway-v0", render_mode="rgb_array")
    # 【重要】collect_demo.pyでの学習データ収集時と全く同じ環境設定
    # （lanes_count / vehicles_density を含む）を使うこと。
    # ここがズレると、学習時とは異なる交通密度・車線数で評価することになり、
    # 「モデルの実力」ではなく「訓練/推論間の分布のズレ」を見てしまう
    # （train-test distribution mismatchの典型的な事故）。
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": config.TARGET_SPEEDS_MPS,
        },
        "vehicles_count": config.VEHICLES_COUNT,
        "lanes_count": config.LANES_COUNT,
        "vehicles_density": config.VEHICLES_DENSITY,
        "duration": MAX_STEPS_PER_EPISODE,
        "policy_frequency": 5,
    })

    feature_dim = int(np.prod(env.observation_space.shape))
    print(f"[main] 観測の形状: {env.observation_space.shape} "
          f"→ flatten後の feature_dim = {feature_dim}")

    model = TrajectoryTransformer(
        feature_dim=feature_dim,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_actions=NUM_ACTIONS,
        max_seq_len=SEQ_LEN,
    )
    print(f"[main] TrajectoryTransformer 初期化完了 "
          f"(d_model={D_MODEL}, nhead={N_HEAD}, num_layers={NUM_LAYERS})")

    # ここが今回の変更点：学習済み重みがあれば読み込む
    is_trained = load_model_weights_if_available(model)

    rss_params = RSSParameters()
    rss_checker = RSSChecker(rss_params)
    print(f"[main] RSSChecker 初期化完了 (params={rss_params})")

    # 全エピソードを通じた行動分布（クラス不均衡対策の効果を確認するための集計）
    overall_action_counter = new_action_counter()

    for episode in range(NUM_EPISODES):
        episode_seed = SEED + episode
        obs, info = env.reset(seed=episode_seed)

        history = deque(maxlen=SEQ_LEN)
        pressure_streak = 0                    # SLOWER提案 or RSS介入が連続した回数
        lane_change_lock_direction = None      # 進行中の車線変更方向(-1:左 / +1:右)
        lane_change_lock_steps_left = 0        # ロックの残りステップ数
        lane_change_lock_target_lane = None    # ロック開始時に狙った車線ID

        print(f"\n===== エピソード {episode + 1}/{NUM_EPISODES} 開始 "
              f"(seed={episode_seed}, "
              f"モデル={'学習済み' if is_trained else '未学習(ランダム)'}) =====")

        total_reward = 0.0
        action_sequence_log = []  # このエピソードのfinal_actionを全部記録する
        rss_intervention_count = 0
        action_counter = new_action_counter()

        for step in range(MAX_STEPS_PER_EPISODE):
            obs_flat = flatten_observation(obs)
            history.append(obs_flat)

            proposed_action, reward_score = propose_action_by_model(
                model, history, SEQ_LEN, feature_dim
            )
            action_counter[ACTION_NAMES[proposed_action]] += 1

            ego = env.unwrapped.vehicle

            # --- 車線変更のラッチ（変更なし） ---
            if lane_change_lock_direction is not None:
                lane_change_lock_steps_left -= 1
                opposite_action = (
                    ACTION_LANE_RIGHT if lane_change_lock_direction == -1
                    else ACTION_LANE_LEFT
                )
                if proposed_action == opposite_action:
                    proposed_action = ACTION_IDLE  # 逆方向の提案だけ無効化する
                if lane_change_lock_steps_left <= 0:
                    lane_change_lock_direction = None

            front_info = get_front_vehicle_info(env)
            if front_info is not None:
                distance, v_rear, v_front = front_info
                was_safe = rss_checker.is_safe(distance, v_rear, v_front)
            else:
                distance, v_rear, v_front, was_safe = None, None, None, True

            # --- 無意味な減速の抑制：十分な車間があるのにSLOWERを
            # 提案し続けるのは、学習データの「減速が連続するシーン」を
            # 引きずった"惰性"であり、安全上の理由ではない。
            # 前車がいない、または十分離れているなら、SLOWER提案は
            # 無視してIDLEとして扱う（RSSの安全判定には一切影響しない）---
            if (
                proposed_action == ACTION_SLOWER
                and (front_info is None or distance > config.RESUME_CLEAR_GAP_M)
            ):
                proposed_action = ACTION_IDLE

            danger_signal = not was_safe
            pressure_streak = pressure_streak + 1 if danger_signal else 0

            # 「今まさに危険」または「危険信号がNUDGE_TRIGGER_STREAK回連続」
            # のどちらかで、検証済みの回避車線を探す。
            # 前者はブレーキが間に合わない緊急時、後者は早期の後押し。
            need_evasive_option = (not was_safe) or (
                pressure_streak >= config.NUDGE_TRIGGER_STREAK
            )

            lane_change_candidate = None
            if need_evasive_option and lane_change_lock_direction is None:
                for direction, lane_action in ((-1, ACTION_LANE_LEFT), (1, ACTION_LANE_RIGHT)):
                    if is_adjacent_lane_clear(env, direction, config.ADJACENT_LANE_CLEAR_MARGIN_M):
                        lane_change_candidate = lane_action
                        if pressure_streak >= config.NUDGE_TRIGGER_STREAK:
                            print(f"  [Overtake Nudge] {config.NUDGE_TRIGGER_STREAK}"
                                  f"ステップ連続の危険信号を検知、車線変更を提案")
                        else:
                            print(f"  [Evasive] RSS危険域を検知、回避のため車線変更を提案")
                        break

            effective_proposed_action = (
                lane_change_candidate if lane_change_candidate is not None else proposed_action
            )

            if (
                effective_proposed_action in (ACTION_LANE_LEFT, ACTION_LANE_RIGHT)
                and lane_change_lock_direction is None
            ):
                lane_change_lock_direction = (
                    -1 if effective_proposed_action == ACTION_LANE_LEFT else 1
                )
                lane_change_lock_steps_left = config.LANE_CHANGE_HOLD_STEPS
                lane_change_lock_target_lane = ego.lane_index[2] + lane_change_lock_direction

            # --- RSSは最終防衛ラインとして必ず安全判定を行う（スキップしない） ---
            if front_info is not None:
                final_action = rss_checker.override_action(
                    current_distance=distance,
                    v_rear=v_rear,
                    v_front=v_front,
                    proposed_action=effective_proposed_action,
                    brake_action=ACTION_SLOWER,
                    lane_change_action=lane_change_candidate,
                )
                if not was_safe:
                    rss_intervention_count += 1
            else:
                final_action = effective_proposed_action

            action_sequence_log.append(final_action)

            obs, reward, terminated, truncated, info = env.step(final_action)
            total_reward += reward

            if terminated or truncated:
                print(f"  → ステップ{step}でエピソード終了 "
                      f"(terminated={terminated}, truncated={truncated})")
                break

        # ------------------------------------------------------------
        # エピソード内で、上位プランナーがどの行動をどれだけ提案したかを
        # 集計して表示する。未学習時は「SLOWERばかり」、学習後は
        # 「FASTERやLANE_LEFT/RIGHTも織り交ぜる」といった変化が
        # ここで一目で分かるようになっている。
        # ------------------------------------------------------------
        print(f"===== エピソード {episode + 1} 終了："
              f"累積報酬={total_reward:.2f}, "
              f"RSS介入回数={rss_intervention_count}回 =====")
        print(format_action_distribution(
            action_counter, f"エピソード{episode + 1}の行動分布"
        ))

        import hashlib
        seq_hash = hashlib.md5(str(action_sequence_log).encode()).hexdigest()[:8]
        print(f"  [再現性チェック用ハッシュ] {seq_hash}")

        merge_action_counters(overall_action_counter, action_counter)

    env.close()

    # ------------------------------------------------------------
    # 全エピソード合計の行動分布を表示する。
    # 【使い方】学習前（未学習/旧モデル）と学習後（再学習済みモデル）の
    # それぞれでこのスクリプトを実行し、ここの数値・比率を見比べることで
    # 「クラス不均衡対策によってLANE_LEFT/RIGHTの選択率が
    #  約4% → 何%まで改善したか」を定量的に示すことができる。
    # ------------------------------------------------------------
    print("\n" + "=" * 88)
    print(format_action_distribution(overall_action_counter, "全エピソード合計の行動分布"))
    print("=" * 88)


if __name__ == "__main__":
    run_simulation()