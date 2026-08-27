"""
collect_demo.py
=====================================================================
【概要】
highway-env に標準搭載されている IDM (Intelligent Driver Model) と
MOBIL（車線変更モデル）を「エキスパート」として利用し、
「スムーズな追従走行」「必要な時だけ車線変更して追い越す」という
お手本となる行動データ（デモンストレーション）を自動収集するスクリプト。
 
【エキスパートの作り方（重要な設計判断）】
highway-envには「離散行動（LANE_LEFT/IDLE/LANE_RIGHT/FASTER/SLOWER）を
選ぶエキスパート」は標準では用意されていない。
一方、highway-env内部で他車の運転に使われている IDMVehicle クラスは、
「前方車両との距離・速度をセンシングし、連続値の加速度指令とMOBILモデルに
基づく車線変更判断を自動で行う」という非常によく出来たルールベース方策を
既に持っている。
 
そこで本スクリプトでは、
    1. 現在の自車(ego)とまったく同じ位置・速度を持つ「影武者」の
       IDMVehicleインスタンス(shadow)を、毎ステップその場で仮想的に作る
    2. shadow.act() を呼び出し、IDM+MOBILに「今この瞬間、何をすべきか」
       （車線変更するか／どれくらい加速・減速すべきか）を計算させる
    3. その結果を、私たちが使っている5つの離散行動ラベルに変換する
    4. そのラベル（＝エキスパートの行動）をそのまま実際の環境に適用し、
       egoを実際に走行させる
という手順で、「highway-env組み込みの高品質なルールベース方策」を
そのまま離散行動のエキスパートラベルへ変換している。
 
なお、この shadow インスタンスは road.vehicles には追加していないため、
実際のシミュレーション上の他車の挙動には一切影響を与えない。
あくまで「今この瞬間、IDM+MOBILならどう判断するか」を覗き見るためだけの
使い捨てオブジェクトである。
 
【衝突エピソードの扱い】
このシミュレーションはあくまでIDM+MOBILの近似的な変換であり、まれに
渋滞などで衝突（terminated=True）が発生することがある。
「衝突に至った運転」はお手本として不適切なため、該当エピソードのデータは
丸ごと破棄している。
=====================================================================
"""

# 必要なライブラリをインポート
import random
from collections import deque

import numpy as np
import torch

import gymnasium as gym
import highway_env
from highway_env.vehicle.behavior import IDMVehicle
# 自作のフォルダをインポート
import config
from seed_utils import set_global_seed
from transformer_planner import ACTION_NAMES
from main import flatten_observation, build_input_tensor


# =====================================================================
# 行動ID定数（highway-envのDiscreteMetaActionと対応。main.pyと同じ定義）
# =====================================================================
ACTION_LANE_LEFT = 0
ACTION_IDLE = 1
ACTION_LANE_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4
 
# 縦方向の加速度指令をFASTER/IDLE/SLOWERへ変換する際の閾値 [m/s^2]
# |acceleration| がこの値未満なら「現状維持(IDLE)」とみなす
ACCEL_THRESHOLD = 0.3
 
# ------------------------------------------------------------------
# 「影武者」IDMVehicleに与える"巡航希望速度" [m/s]
#
# 【重要な設計判断（ハマりやすい罠）】
# 当初は ego.target_speed （＝egoが過去にSLOWERを選ぶたびに1段階ずつ
# 下がっていく、離散化された目標速度）をそのままIDMの目標速度として
# 渡していた。しかし、これには重大な問題があった：
#   一度前方車両との接近でSLOWERを選ぶ
#     → ego.target_speedが1段階下がる
#     → 次の判断でもIDMの基準速度が下がったままなので、
#       前方が空いていてもなかなか正の加速度(FASTER)が出ない
#     → 目標速度が下がったままになり、二度と加速に転じない
#       "ラチェット現象"に陥ってしまう
#
# 実際にhighway-env内の他車(IDMVehicle)は、meta-actionの影響を受けない
# "各車両ごとに固定された巡航希望速度"を持ち続けている。
# そこで本実装でも、ego自身のtarget_speedではなく、
# 「このドライバーが本来目指したい一定の巡航速度」を表す定数を
# 別途用意し、それをIDMの計算に使うことで、
# 「前が空けば加速して巡航速度に戻り、詰まれば減速する」という
# 自然な追従・追い越し挙動を再現できるようにしている。
# ------------------------------------------------------------------
EGO_DESIRED_CRUISE_SPEED = 25.0  # [m/s] このエキスパートが目指す巡航速度

def get_expert_action(env: gym.Env) -> tuple[int, float]:
    """
    現在のego車両の状態から「IDM+MOBILならどう判断するか」を計算し、
    離散行動ラベルに変換して返す関数。
 
    Parameters
    ----------
    env : gym.Env
        highway-envの環境インスタンス
 
    Returns
    -------
    tuple(int, float)
        (エキスパートが選ぶ行動ID, IDMが計算した生の加速度指令値[m/s^2]（ログ用）)
    """
    ego = env.unwrapped.vehicle
    road = ego.road
 
    # ------------------------------------------------------------
    # (1) egoと全く同じ運動状態を持つ「影武者」のIDMVehicleを生成する。
    #
    #     timer には意図的に IDMVehicle.LANE_CHANGE_DELAY 以上の値を渡す。
    #     本来のIDMVehicleは「経過時間がLANE_CHANGE_DELAY[s]を超えるたびに
    #     車線変更を再検討する」という間欠的な判断をするが、
    #     毎回新しいshadowを作り直す本実装ではその経過時間の概念がない。
    #     そこで「常に判断可能な状態」を強制することで、
    #     "今この瞬間、車線変更すべきか"を都度シンプルに問い合わせている。
    #     （実際のhighway-env車両より車線変更の判断頻度がやや高くなる
    #       近似だが、デモ収集用の簡略化として許容する）
    # ------------------------------------------------------------
    # 内部では utils.do_every(duration, timer) が "duration < timer" で
    # 判定されるため、timer をちょうど LANE_CHANGE_DELAY と同じ値にすると
    # 「等しい場合はFalse」と判定されて評価がスキップされてしまう。
    # そのため、確実に評価を発火させるよう余裕を持った値を渡す。
    forced_timer = IDMVehicle.LANE_CHANGE_DELAY + 1.0
 
    shadow = IDMVehicle(
        road=road,
        position=ego.position.copy(),
        heading=ego.heading,
        speed=ego.speed,
        target_lane_index=ego.target_lane_index,
        # ここが重要な修正点：ego.target_speed（ラチェットする離散値）ではなく、
        # 固定の巡航希望速度を渡す（詳細は上のコメント参照）
        target_speed=EGO_DESIRED_CRUISE_SPEED,
        route=ego.route,
        enable_lane_change=True,
        timer=forced_timer,  # 強制的にMOBIL評価を発火させる
    )
 
    # ------------------------------------------------------------
    # (2) IDM+MOBILの判断を実行させる。
    #     内部で self.target_lane_index（車線変更したいか）と
    #     self.action["acceleration"]（縦方向の加速度指令）が設定される。
    #     ※ act()は"意思決定"のみを行い、実際には位置を動かさない
    #       （位置の更新は step() の役割であり、ここでは呼び出していない
    #        ため、shadowは実際のシミュレーションに一切影響を与えない）
    #
    #     【重要なバグ対策】
    #     shadowはegoと全く同じposition（座標）を持っているため、
    #     road.neighbour_vehicles() が「shadowのすぐ前にいる車」を
    #     探索する際、"ego自身"を距離0の前方車両として誤検知してしまう
    #     （shadowとegoは別オブジェクトなので "v is vehicle" の
    #      同一性チェックをすり抜けてしまうため）。
    #     この自己干渉を避けるため、act()を呼ぶ間だけ一時的に
    #     egoをroad.vehiclesから取り除き、処理が終わったら必ず元に戻す。
    # ------------------------------------------------------------
    ego_was_removed = False
    if ego in road.vehicles:
        road.vehicles.remove(ego)
        ego_was_removed = True
 
    try:
        shadow.act()
    finally:
        # try/finallyにすることで、途中で例外が発生してもegoが
        # road.vehiclesから消えたままにならないよう保証している
        if ego_was_removed:
            road.vehicles.append(ego)
 
    # ------------------------------------------------------------
    # (3-a) 横方向の判断：車線変更するかどうか
    #     lane_index / target_lane_index はどちらも
    #     (from_node, to_node, lane_id) の3要素タプルであり、
    #     3番目の要素(lane_id)が車線番号を表す。
    #     highway-envの規約では、lane_idが小さいほど左車線、
    #     大きいほど右車線となる。
    # ------------------------------------------------------------
    if shadow.target_lane_index[2] < ego.lane_index[2]:
        action_id = ACTION_LANE_LEFT
    elif shadow.target_lane_index[2] > ego.lane_index[2]:
        action_id = ACTION_LANE_RIGHT
    else:
        # ------------------------------------------------------------
        # (3-b) 車線変更しない場合は、縦方向の加速度指令を
        #       FASTER / IDLE / SLOWER のいずれかに変換する
        # ------------------------------------------------------------
        accel = shadow.action["acceleration"]
        if accel > ACCEL_THRESHOLD:
            action_id = ACTION_FASTER
        elif accel < -ACCEL_THRESHOLD:
            action_id = ACTION_SLOWER
        else:
            action_id = ACTION_IDLE
 
    raw_acceleration = float(shadow.action["acceleration"])
    return action_id, raw_acceleration


# =====================================================================
# クラス不均衡対策①：意図的な「遅い前走車」の注入
# =====================================================================
def inject_slow_lead_vehicles(env: gym.Env, rng: random.Random) -> int:
    """
    reset() 直後、および以後 config.SLOW_LEAD_REINJECT_INTERVAL_STEPS
    ステップおきに呼び出すことを想定した関数。
    egoの前方かつ近い車線（左隣・同一車線・右隣）にいる車両のうち、
    egoに近い順に最大 config.SLOW_LEAD_MAX_COUNT 台の巡航速度を
    強制的に引き下げる。

    【なぜ必要か】
    highway-envの他車(IDMVehicle)の初期速度は基本的にランダムサンプリング
    されるため、「前が遅くて追い越すしかない」という状況が自然には
    十分な頻度で発生しない。これが、エキスパートデータにおける
    LANE_LEFT/LANE_RIGHTサンプルが約4%しかなかった根本原因の一つ。
    そこで、明示的に「前走車が遅い」シーンを一定確率で作り出すことで、
    IDM+MOBILエキスパートが「追い越すべきだ」と判断する場面そのものを
    増やす（＝手本となる車線変更データの絶対数を増やす）。

    reset直後の1回だけでなく一定間隔で再度呼び出すことで、
    1エピソード（最大100ステップ）の中で複数回「追い越すべき」
    シーンが発生するようになり、オーバーサンプリング（＝複製）に
    頼らなくても本物の車線変更サンプルの絶対数を稼げるようにしている。

    Parameters
    ----------
    env : gym.Env
        reset() 済みのhighway-env環境
    rng : random.Random
        このエピソード専用に切られた乱数生成器。
        グローバルな乱数状態（set_global_seedで固定した状態）を
        汚さないよう、呼び出し側で `random.Random(episode_seed)` の
        ように独立したインスタンスを渡すこと。

    Returns
    -------
    int
        実際に減速させた車両の台数（ログ・デバッグ用）
    """
    # 確率的にスキップする（＝介入なしの"普通の"シーンも一定割合残し、
    # データの多様性を保つ）
    if rng.random() > config.SLOW_LEAD_INJECTION_PROB:
        return 0

    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
    ego_lane = ego.lane_index[2]

    # ego前方・近接車線（左隣/同一/右隣）にいる車両を候補として集める
    candidates = [
        v for v in road.vehicles
        if v is not ego
        and abs(v.lane_index[2] - ego_lane) <= 1
        and 0.0 < (v.position[0] - ego.position[0]) <= config.SLOW_LEAD_DETECT_RANGE_M
    ]
    # egoに近い順（＝最も直接的に走行を妨げる車両から優先）にソート
    candidates.sort(key=lambda v: v.position[0] - ego.position[0])

    slowed_count = 0
    for v in candidates[: config.SLOW_LEAD_MAX_COUNT]:
        slow_speed = rng.uniform(*config.SLOW_LEAD_SPEED_RANGE)
        v.speed = slow_speed
        # IDMVehicleは target_speed を「本来目指したい巡航速度」として
        # 保持し続けるため、ここも合わせて下げないと数ステップで
        # 元の速度に加速し戻ってしまい、介入の効果が消えてしまう
        if hasattr(v, "target_speed"):
            v.target_speed = slow_speed
        slowed_count += 1

    return slowed_count


# =====================================================================
# クラス不均衡対策②：収集後のオーバーサンプリング
# =====================================================================
def oversample_lane_change_samples(
    obs_windows: list, actions: list, target_ratio: float
) -> tuple:
    """
    収集済みの (観測ウィンドウ, 行動ラベル) データにおいて、
    LANE_LEFT / LANE_RIGHT の比率が target_ratio に満たない場合、
    それらのサンプルを複製して比率を底上げするオーバーサンプリング関数。

    【複製時にノイズを加える理由】
    観測ウィンドウを寸分違わず複製すると、学習データ中に「全く同一の
    入力」が何度も出現することになり、その一部だけを過剰に記憶して
    しまう（過学習）リスクがわずかに高まる。
    そこで複製のたびに、config.LANE_CHANGE_OVERSAMPLE_NOISE_STD 程度の
    ごく小さいガウスノイズを観測ベクトルに加算する。これは画像分野で
    いう Data Augmentation に近い発想で、「意味的にはほぼ同じ状況だが、
    数値としては微妙に異なるサンプル」を作り出すことで、単純な暗記を防ぐ。

    Parameters
    ----------
    obs_windows : list[torch.Tensor]
        各要素 shape=[SEQ_LEN, feature_dim]
    actions : list[int]
        obs_windowsと対応する行動ラベル
    target_ratio : float
        LANE_LEFT+LANE_RIGHTが全体に占めるべき目標比率（例: 0.18）

    Returns
    -------
    tuple(list, list)
        オーバーサンプリング後の (obs_windows, actions)。
        目標比率にすでに達している場合、または複製元のサンプルが
        1件も無い場合は、入力をそのまま返す。
    """
    lane_change_ids = (ACTION_LANE_LEFT, ACTION_LANE_RIGHT)
    lane_change_indices = [i for i, a in enumerate(actions) if a in lane_change_ids]

    n_total = len(actions)
    n_lane_change = len(lane_change_indices)
    current_ratio = n_lane_change / max(1, n_total)

    if current_ratio >= target_ratio:
        print(f"[collect_demo] 車線変更比率は既に目標({target_ratio*100:.1f}%)"
              f"を満たしています（現在{current_ratio*100:.1f}%）。"
              f"オーバーサンプリングは行いません。")
        return obs_windows, actions

    if n_lane_change == 0:
        print("[collect_demo][警告] 車線変更サンプルが1件も収集できなかった"
              "ため、オーバーサンプリングできません。"
              "config.SLOW_LEAD_INJECTION_PROB を上げるか、"
              "EXTRA_COLLECTION_ROUNDS を増やして再収集してください。")
        return obs_windows, actions

    # target_ratio = (n_lane_change + x) / (n_total + x) をxについて解く
    # → x = (target_ratio * n_total - n_lane_change) / (1 - target_ratio)
    required_extra = (target_ratio * n_total - n_lane_change) / (1.0 - target_ratio)
    required_extra = max(0, int(np.ceil(required_extra)))

    # 際限のない複製を防ぐガード（元の車線変更サンプル数の何倍まで許すか）
    max_extra = n_lane_change * config.LANE_CHANGE_OVERSAMPLE_MAX_MULTIPLIER
    capped = required_extra > max_extra
    n_extra = min(required_extra, max_extra)

    noise_std = config.LANE_CHANGE_OVERSAMPLE_NOISE_STD
    extra_obs = []
    extra_actions = []
    for i in range(n_extra):
        # 元の車線変更サンプルを均等に使い回しながら複製する
        src_idx = lane_change_indices[i % n_lane_change]
        window = obs_windows[src_idx].clone()
        noise = torch.randn_like(window) * noise_std
        extra_obs.append(window + noise)
        extra_actions.append(actions[src_idx])

    new_obs_windows = obs_windows + extra_obs
    new_actions = actions + extra_actions
    new_ratio = (n_lane_change + n_extra) / len(new_actions)

    print(f"[collect_demo] オーバーサンプリング実行: "
          f"{current_ratio*100:.1f}% → {new_ratio*100:.1f}% "
          f"（+{n_extra}サンプル、ノイズstd={noise_std}）")
    if capped:
        print(f"[collect_demo][警告] LANE_CHANGE_OVERSAMPLE_MAX_MULTIPLIER"
              f"({config.LANE_CHANGE_OVERSAMPLE_MAX_MULTIPLIER}倍)の上限に達したため、"
              f"目標比率({target_ratio*100:.1f}%)には届いていません。"
              f"根本的に改善するには、収集エピソード数や"
              f"SLOW_LEAD_INJECTION_PROBを上げてください。")

    return new_obs_windows, new_actions


def _run_collection_episodes(
    env: gym.Env,
    feature_dim: int,
    num_episodes: int,
    seed_start: int,
) -> tuple:
    """
    指定エピソード数分のデモ収集を実行する内部ヘルパー関数。
    collect_demonstrations() の初回収集・追加収集ラウンドの両方から
    共通して呼び出される（コード重複を避けるための関数化）。

    Parameters
    ----------
    env : gym.Env
    feature_dim : int
    num_episodes : int
        収集するエピソード数
    seed_start : int
        このバッチで使う最初のエピソードシード
        （config.SEED + 累積エピソード数、というように呼び出し側で
        毎回ズラして渡すことで、ラウンドをまたいでシードが重複しないようにする）

    Returns
    -------
    tuple(list, list, dict, int)
        (obs_windows, actions, action_counter, collected_episodes)
    """
    obs_windows = []
    actions = []
    action_counter = {name: 0 for name in ACTION_NAMES}
    collected_episodes = 0

    for i in range(num_episodes):
        episode_seed = seed_start + i
        obs, info = env.reset(seed=episode_seed)

        # このエピソード専用の独立した乱数生成器
        # （set_global_seedで固定したPython標準randomの状態には触れず、
        #   常にエピソードシードから決定的に振る舞いを再現できるようにする）
        episode_rng = random.Random(episode_seed)
        total_slowed = inject_slow_lead_vehicles(env, episode_rng)
        reinject_count = 1  # reset直後の1回をカウント済みとして扱う

        history = deque(maxlen=config.SEQ_LEN)
        episode_obs_windows = []
        episode_actions = []
        crashed = False

        for step in range(config.MAX_STEPS_PER_EPISODE):
            # --------------------------------------------------------
            # 【重要】reset直後の1回だけの介入だと、最初の追い越しが
            # 終わった後の残り70〜90ステップは"平坦なIDLE走行"に戻って
            # しまい、1エピソードあたりの本物の車線変更サンプル数が
            # 頭打ちになる。そこで config.SLOW_LEAD_REINJECT_INTERVAL_STEPS
            # ステップおきに再度「遅い前走車」判定を行い、1エピソード内で
            # 複数回、追い越すべきシーンを作り出す。
            # （episode_rngの内部確率でスキップされることも多いため、
            #   毎回必ず介入するわけではない点に注意）
            # --------------------------------------------------------
            if (
                step > 0
                and step % config.SLOW_LEAD_REINJECT_INTERVAL_STEPS == 0
            ):
                slowed_now = inject_slow_lead_vehicles(env, episode_rng)
                if slowed_now > 0:
                    total_slowed += slowed_now
                    reinject_count += 1

            obs_flat = flatten_observation(obs)
            history.append(obs_flat)

            expert_action, raw_accel = get_expert_action(env)

            window_tensor = build_input_tensor(history, config.SEQ_LEN, feature_dim)
            episode_obs_windows.append(window_tensor.squeeze(0))
            episode_actions.append(expert_action)

            obs, reward, terminated, truncated, info = env.step(expert_action)

            if terminated:
                crashed = True
                break
            if truncated:
                break

        if not crashed:
            obs_windows.extend(episode_obs_windows)
            actions.extend(episode_actions)
            for a in episode_actions:
                action_counter[ACTION_NAMES[a]] += 1
            collected_episodes += 1
            print(f"  episode seed={episode_seed:4d}: "
                  f"{len(episode_actions):3d}ステップ収集（クラッシュなし、"
                  f"減速介入={total_slowed}台/{reinject_count}回）")
        else:
            print(f"  episode seed={episode_seed:4d}: 衝突のため破棄")

    return obs_windows, actions, action_counter, collected_episodes


def _print_action_distribution(action_counter: dict, total: int, title: str) -> None:
    """行動の内訳（件数・比率）を整形して表示する共通ヘルパー。"""
    print(f"[collect_demo] {title}")
    for name, count in action_counter.items():
        ratio = 100.0 * count / max(1, total)
        print(f"    {name:>10s}: {count:5d}件 ({ratio:5.1f}%)")
    lane_change_count = (
        action_counter.get("LANE_LEFT", 0) + action_counter.get("LANE_RIGHT", 0)
    )
    lane_change_ratio = 100.0 * lane_change_count / max(1, total)
    print(f"    ---- 車線変更(LANE_LEFT+LANE_RIGHT)合計: "
          f"{lane_change_count:5d}件 ({lane_change_ratio:5.1f}%) ----")


def collect_demonstrations() -> None:
    """
    複数エピソードにわたってデモンストレーションを収集し、
    (観測ウィンドウ, 行動ラベル) のペアを config.DEMO_DATA_PATH に保存する。

    【第4段階での変更点：クラス不均衡対策】
        (1) 環境設定に LANES_COUNT / VEHICLES_DENSITY を追加し、
            交通量そのものを増やして意思決定の機会を増やす
        (2) 各エピソードの reset() 直後に、確率的に前走車を減速させる
            （inject_slow_lead_vehicles）ことで、「追い越すべき」シーンを
            意図的に増やす
        (3) 収集後の車線変更比率が config.MIN_LANE_CHANGE_RATIO に届かない
            場合、追加エピソードを収集する（最大 EXTRA_COLLECTION_ROUNDS 回）
        (4) それでも届かない場合は、最終手段としてオーバーサンプリングで
            比率を底上げする
    """
    set_global_seed(config.SEED)

    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "observation": {"type": "Kinematics"},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": config.TARGET_SPEEDS_MPS,
        },
        "vehicles_count": config.VEHICLES_COUNT,
        "lanes_count": config.LANES_COUNT,
        "vehicles_density": config.VEHICLES_DENSITY,
        "duration": config.MAX_STEPS_PER_EPISODE,
        "policy_frequency": 5,
    })

    feature_dim = int(np.prod(env.observation_space.shape))
    print(f"[collect_demo] feature_dim = {feature_dim}")
    print(f"[collect_demo] vehicles_count={config.VEHICLES_COUNT}, "
          f"lanes_count={config.LANES_COUNT}, "
          f"vehicles_density={config.VEHICLES_DENSITY}")
    print(f"[collect_demo] 遅い前走車の注入確率={config.SLOW_LEAD_INJECTION_PROB*100:.0f}%, "
          f"速度レンジ={config.SLOW_LEAD_SPEED_RANGE} m/s")
    print(f"[collect_demo] {config.NUM_DEMO_EPISODES}エピソード分の"
          f"デモンストレーション収集を開始します...\n")

    # ------------------------------------------------------------
    # ① 初回収集ラウンド
    # ------------------------------------------------------------
    all_obs_windows, all_actions, action_counter, collected_episodes = (
        _run_collection_episodes(
            env, feature_dim, config.NUM_DEMO_EPISODES, seed_start=config.SEED
        )
    )
    next_seed = config.SEED + config.NUM_DEMO_EPISODES

    if len(all_actions) == 0:
        env.close()
        raise RuntimeError(
            "収集できたデータが0件でした。config.NUM_DEMO_EPISODESを"
            "増やすか、環境設定を見直してください。"
        )

    _print_action_distribution(
        action_counter, len(all_actions), "初回収集後の行動内訳（データの偏り確認）:"
    )

    # ------------------------------------------------------------
    # ② 車線変更比率が目標に届かなければ、追加エピソードを収集する
    #    （オーバーサンプリングだけに頼ると"見た目の比率"は上がっても
    #      実際に収集できたバリエーションは増えないため、
    #      まずは"本物のデータ"を追加で稼ぐことを優先する）
    # ------------------------------------------------------------
    for round_idx in range(1, config.EXTRA_COLLECTION_ROUNDS + 1):
        lane_change_count = (
            action_counter.get("LANE_LEFT", 0) + action_counter.get("LANE_RIGHT", 0)
        )
        current_ratio = lane_change_count / max(1, len(all_actions))
        if current_ratio >= config.MIN_LANE_CHANGE_RATIO:
            break

        print(f"\n[collect_demo] 車線変更比率({current_ratio*100:.1f}%)が"
              f"目標({config.MIN_LANE_CHANGE_RATIO*100:.1f}%)未満のため、"
              f"追加収集ラウンド {round_idx}/{config.EXTRA_COLLECTION_ROUNDS} を実行します "
              f"(+{config.EXTRA_COLLECTION_EPISODES_PER_ROUND}エピソード)...")

        extra_obs, extra_actions, extra_counter, extra_collected = (
            _run_collection_episodes(
                env,
                feature_dim,
                config.EXTRA_COLLECTION_EPISODES_PER_ROUND,
                seed_start=next_seed,
            )
        )
        next_seed += config.EXTRA_COLLECTION_EPISODES_PER_ROUND

        all_obs_windows.extend(extra_obs)
        all_actions.extend(extra_actions)
        collected_episodes += extra_collected
        for name, count in extra_counter.items():
            action_counter[name] += count

    env.close()

    _print_action_distribution(
        action_counter, len(all_actions),
        "追加収集後（オーバーサンプリング適用前）の行動内訳:",
    )

    # ------------------------------------------------------------
    # ③ それでも目標比率に届かない場合の最終手段：オーバーサンプリング
    # ------------------------------------------------------------
    all_obs_windows, all_actions = oversample_lane_change_samples(
        all_obs_windows, all_actions, target_ratio=config.MIN_LANE_CHANGE_RATIO
    )
    final_counter = {name: 0 for name in ACTION_NAMES}
    for a in all_actions:
        final_counter[ACTION_NAMES[a]] += 1

    # ------------------------------------------------------------
    # 収集したデータをテンソルにまとめて保存する
    # ------------------------------------------------------------
    # all_obs_windows: 長さNのリスト（各要素 shape=[SEQ_LEN, feature_dim]）
    #   → torch.stackで shape=[N, SEQ_LEN, feature_dim] の3階テンソルに変換
    observations_tensor = torch.stack(all_obs_windows, dim=0)

    # all_actions: 長さNのintリスト
    #   → CrossEntropyLossの正解ラベルとして使うため dtype=torch.long にする
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)  # shape=[N]

    print(f"\n[collect_demo] 収集完了：{collected_episodes}エピソード、"
          f"合計{len(all_actions)}サンプル"
          f"（オーバーサンプリング後）")
    print(f"[collect_demo] observations_tensor.shape = "
          f"{tuple(observations_tensor.shape)}  ([N, SEQ_LEN, feature_dim])")
    print(f"[collect_demo] actions_tensor.shape = "
          f"{tuple(actions_tensor.shape)}  ([N])")
    _print_action_distribution(
        final_counter, len(all_actions), "最終的な行動の内訳（学習データとして保存する分布）:"
    )

    # dict形式で保存することで、feature_dimやseq_lenといったメタ情報も
    # 一緒に保存でき、train_irl.py側で設定の整合性チェックに使える
    torch.save(
        {
            "observations": observations_tensor,
            "actions": actions_tensor,
            "feature_dim": feature_dim,
            "seq_len": config.SEQ_LEN,
        },
        config.DEMO_DATA_PATH,
    )
    print(f"\n[collect_demo] デモデータを {config.DEMO_DATA_PATH} に保存しました。")


if __name__ == "__main__":
    collect_demonstrations()