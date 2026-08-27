"""
visualize.py
=====================================================================
【概要】
これまでに作った推論パイプライン（TrajectoryTransformer + RSSChecker）を
実際にhighway-env上で走らせながら、Tesla FSD（Full Self-Driving）の
車載ディスプレイを思わせる、洗練されたミニマルUIを描画画面に
リアルタイムでオーバーレイし、その様子を .mp4 動画として保存する。

【描画パイプラインの考え方（cv2とPILの役割分担）】
    highway-env（rgb_array） … 走行シーンそのもの（3D風の道路・車の描画）
            │
            ▼ そのままだと "データ" が何も乗っていない生の映像
    ┌─────────────────────────┐
    │ PIL(Pillow)でUIレイヤーを重ねる  │
    │  - 角丸半透明カード                │ ← ImageDraw.rounded_rectangle
    │  - アンチエイリアスの効いた文字     │ ← ImageDraw.text (TrueTypeフォント)
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │ cv2(OpenCV)で動画に書き出す      │ ← cv2.VideoWriter
    └─────────────────────────┘

    cv2.putText は速いが文字がギザギザ（アンチエイリアスなし）になりやすく、
    "洗練された"ダッシュボード風の見た目には向かない。
    そこで「本物の走行シーン画像の取得とビデオ書き出し」はcv2に任せつつ、
    「カードや文字といったUIレイヤーの描画」だけはPILに任せることで、
    処理速度と見た目の美しさを両立させている。

【フォントについて】
    数値（速度・車間距離など）には等幅フォント(JetBrains Mono)を使用している。
    可変幅フォントだと「72」→「108」のように桁数が変わるたびに文字の
    横幅が変わり、動画再生時にUIがガタガタ揺れて見えてしまう。
    等幅フォントを使うことで、桁数が変わってもカード内のレイアウトが
    左右にブレず、安定した"計器"らしい見た目になる。

【実行方法】
    python3 visualize.py
    → config.VIS_SEED（=43）で1エピソード分走行し、
      config.VIDEO_OUTPUT_PATH（既定: drive_visualization.mp4）に保存される。
=====================================================================
"""

import os

# highway-envの描画はpygameを内部で使用しており、GUI画面のないサーバー環境では
# "XDG_RUNTIME_DIR is invalid or not set" という警告が出ることがある。
# ここでは（SDLの描画ドライバ自体は変更せず）XDG_RUNTIME_DIRにアクセス可能な
# 一時ディレクトリを明示的に割り当てることで、この警告を解消しつつ、
# rgb_arrayレンダリングが正しく動作する状態を保っている。
# ※ SDL_VIDEODRIVER=dummy にすると警告は消えるが、
#    実際の描画内容まで真っ黒になってしまうため使用しない。
# ※ gym / highway_env をimportする"前"に設定する必要がある点に注意。
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/highway_env_runtime")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)

from collections import deque

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

import gymnasium as gym
import highway_env  # noqa: F401  # "highway-v0"を登録するための副作用インポート

import config
from seed_utils import set_global_seed
from rss_checker import RSSChecker, RSSParameters, is_adjacent_lane_clear
from transformer_planner import TrajectoryTransformer, ACTION_NAMES, NUM_ACTIONS
from main import (
    flatten_observation,
    build_input_tensor,
    propose_action_by_model,
    get_front_vehicle_info,
    load_model_weights_if_available,
    ACTION_LANE_LEFT,
    ACTION_IDLE,
    ACTION_LANE_RIGHT,
    ACTION_SLOWER,
)
from eval_utils import new_action_counter, format_action_distribution


# =====================================================================
# デザイントークン（配色・フォント・余白など、UIの見た目をここに集約する）
# 数値をここでまとめて管理することで、"少し色味を変えたい"
# "文字サイズを調整したい"といった調整がこの一箇所で完結する。
# =====================================================================
# FONT_DIR = "/mnt/skills/examples/canvas-design/canvas-fonts"

COLOR_TEXT_PRIMARY = (25, 25, 25, 255)       # カード内の主要テキスト（ほぼ黒）
COLOR_TEXT_SECONDARY = (120, 120, 120, 255)  # カード内のラベル（グレー）
COLOR_TEXT_ON_DARK = (245, 245, 245, 255)    # 走行シーン上に直接乗せる白文字
COLOR_TEXT_ON_DARK_DIM = (225, 225, 225, 210)

COLOR_CARD_BG = (248, 248, 248, 175)         # 白〜グレーの半透明カード背景
COLOR_CARD_BORDER = (255, 255, 255, 130)     # カードの縁（うっすら光らせる）

COLOR_ACCENT_NORMAL = (0, 168, 232)          # 平常時のアクセント（Tesla風シアン）
COLOR_ACCENT_ALERT = (224, 122, 63)          # RSS介入時のアクセント（控えめなオレンジ）

CARD_RADIUS = 16
CARD_PADDING = 16

import platform

def get_system_font_path(font_name: str) -> str:
    """OSごとの標準フォントディレクトリから該当フォントのパスを取得する"""
    system_name = platform.system()
    if system_name == "Windows":
        return f"C:/Windows/Fonts/{font_name}"
    elif system_name == "Darwin":  # macOS
        return f"/System/Library/Fonts/Supplemental/{font_name}"
    else:  # Linux
        return f"/usr/share/fonts/truetype/dejavu/{font_name}"

def load_fonts() -> dict:
    """
    OS標準フォントを自動検索して読み込む関数。
    見つからない場合は自動的にPillowのデフォルトフォントにフォールバックする。
    """
    # OSごとの標準フォント候補（太字系・レギュラー系・等幅系）
    system_name = platform.system()
    if system_name == "Windows":
        bold_font = "arialbd.ttf"
        reg_font = "arial.ttf"
        mono_font = "consola.ttf"
    elif system_name == "Darwin":  # macOS
        bold_font = "Arial Bold.ttf"
        reg_font = "Arial.ttf"
        mono_font = "Courier New.ttf"
    else:  # Linux
        bold_font = "DejaVuSans-Bold.ttf"
        reg_font = "DejaVuSans.ttf"
        mono_font = "DejaVuSansMono.ttf"

    def safe_load(font_name: str, size: int):
        font_path = get_system_font_path(font_name)
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            try:
                # パス指定なしでシステム検索を試行
                return ImageFont.truetype(font_name, size)
            except OSError:
                # 最終フォールバック（デフォルトフォント）
                return ImageFont.load_default()

    return {
        "speed_hero": safe_load(bold_font, 64),
        "speed_unit": safe_load(reg_font, 20),
        "badge": safe_load(bold_font, 18),
        "card_title": safe_load(bold_font, 15),
        "card_label": safe_load(reg_font, 14),
        "card_value": safe_load(mono_font, 20),
        "card_value_small": safe_load(mono_font, 13),
    }


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    shadow_offset: int = 2,
) -> None:
    """
    走行シーンの上に直接乗せる文字（背景が明るい草地や暗いアスファルトなど
    一定しない）でも視認性を保つため、薄い黒の影を1つ後ろにずらして描いてから
    本体の文字を描く、簡易的な"ドロップシャドウ"付きテキスト描画関数。

    Parameters
    ----------
    draw : ImageDraw.ImageDraw
        描画先のImageDrawオブジェクト
    xy : tuple(int, int)
        描画開始座標
    text : str
        描画する文字列
    font : ImageFont.FreeTypeFont
        使用するフォント
    fill : tuple
        文字本体の色 (R, G, B, A)
    shadow_offset : int, default=2
        影をどれだけずらすか[px]
    """
    x, y = xy
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 140))
    draw.text((x, y), text, font=font, fill=fill)


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple,
    title: str,
    rows: list,
    fonts: dict,
    highlight: bool = False,
    accent_color: tuple = COLOR_ACCENT_NORMAL,
    footer_tag: str = None,
) -> None:
    """
    Tesla風の角丸・半透明な情報カードを1枚描画する関数。

    Parameters
    ----------
    draw : ImageDraw.ImageDraw
        描画先
    box : tuple(int, int, int, int)
        カードの (x0, y0, x1, y1)
    title : str
        カード見出し（例: "SPEED & CONTROL"）
    rows : list[tuple(str, str)]
        カード内に並べる (ラベル, 値) のペアのリスト
        例: [("Speed", "72 km/h"), ("Target", "90 km/h")]
    fonts : dict
        load_fonts()で読み込んだフォント辞書
    highlight : bool, default=False
        Trueの場合、カードの縁をアクセントカラーで強調表示する
        （RSS介入時など、注意を引きたいときに使う）
    accent_color : tuple, default=COLOR_ACCENT_NORMAL
        highlight=Trueのときに使う強調色
    footer_tag : str, optional
        カード右下に小さく表示するタグ文字列（例: "[ OVERRIDE ]"）。
        指定した場合、accent_colorで着色して表示する。
    """
    x0, y0, x1, y1 = box

    # --- カード本体（角丸・半透明の背景） ---
    border_color = (*accent_color, 220) if highlight else COLOR_CARD_BORDER
    border_width = 2 if highlight else 1
    draw.rounded_rectangle(
        box, radius=CARD_RADIUS, fill=COLOR_CARD_BG, outline=border_color, width=border_width
    )

    # highlight時は、カード上端にアクセントカラーの細いバーを追加して
    # 「このカードが今、特別な状態にある」ことを一目で分かるようにする
    if highlight:
        draw.rounded_rectangle(
            (x0 + 10, y0 - 3, x1 - 10, y0 + 6),
            radius=4,
            fill=(*accent_color, 235),
        )

    # --- 見出し（左）＋ 右上の小さなステータスタグ（例: "[ OVERRIDE ]"） ---
    text_x = x0 + CARD_PADDING
    text_y = y0 + CARD_PADDING
    title_color = (*accent_color, 255) if highlight else COLOR_TEXT_SECONDARY
    draw.text((text_x, text_y), title.upper(), font=fonts["card_title"], fill=title_color)

    if footer_tag:
        # タグは値の行と縦方向で衝突しないよう、見出しと同じ行の右端に配置する
        tag_w = draw.textlength(footer_tag, font=fonts["card_value_small"])
        draw.text(
            (x1 - CARD_PADDING - tag_w, text_y + 1),
            footer_tag,
            font=fonts["card_value_small"],
            fill=(*accent_color, 255),
        )

    # --- ラベル/値のペアを縦に並べる ---
    row_y = text_y + 28
    for label, value in rows:
        draw.text((text_x, row_y), label, font=fonts["card_label"], fill=COLOR_TEXT_SECONDARY)
        # 値は右寄せにすることで、"計器パネル"らしい整列にする
        value_w = draw.textlength(value, font=fonts["card_value"])
        draw.text(
            (x1 - CARD_PADDING - value_w, row_y - 3),
            value,
            font=fonts["card_value"],
            fill=(*accent_color, 255) if highlight else COLOR_TEXT_PRIMARY,
        )
        row_y += 26


def render_ui_overlay(frame_rgb: np.ndarray, telemetry: dict, fonts: dict) -> np.ndarray:
    """
    highway-envが描画した1フレーム(RGB画像)に、Tesla FSD風のUIレイヤーを
    重ね合わせる関数。

    Parameters
    ----------
    frame_rgb : np.ndarray
        shape = [H, W, 3]、highway-envのenv.render()が返す生の走行シーン画像
    telemetry : dict
        その瞬間の車両状態・AIの判断結果をまとめた辞書。
        必須キー: speed_kmh, target_speed_kmh, ai_plan, final_action,
                  gap_m, margin_m, rss_active, irl_reward
    fonts : dict
        load_fonts()で読み込んだフォント辞書

    Returns
    -------
    np.ndarray
        shape = [H, W, 3]、UIが重ね合わされたRGB画像
    """
    h, w, _ = frame_rgb.shape

    # --------------------------------------------------------------
    # PILでの合成は「不透明の背景画像」+「透明な描画レイヤー」を
    # alpha_compositeで合成する、という2枚構成で行う。
    # こうすることで、カードの半透明処理（アルファブレンド）を
    # PIL側の高品質なアンチエイリアス処理に任せられる。
    # --------------------------------------------------------------
    base_img = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    is_rss_active = telemetry["rss_active"]
    accent = COLOR_ACCENT_ALERT if is_rss_active else COLOR_ACCENT_NORMAL

    # ================================================================
    # (1) 上部：速度のヒーロー表示 + モードバッジ
    # ================================================================
    # 走行シーンの明るさが場所によって変わっても文字が読めるよう、
    # 影付きテキスト(draw_text_with_shadow)を使う
    speed_text = f"{telemetry['speed_kmh']:.0f}"
    draw_text_with_shadow(
        draw, (28, 18), speed_text, fonts["speed_hero"], COLOR_TEXT_ON_DARK
    )
    speed_text_w = draw.textlength(speed_text, font=fonts["speed_hero"])
    draw_text_with_shadow(
        draw, (28 + speed_text_w + 10, 52), "km/h", fonts["speed_unit"], COLOR_TEXT_ON_DARK_DIM
    )

    # --- モードバッジ（右上）：[ DRIVE ] / [ RSS ACTIVE ] ---
    mode_text = "[ RSS ACTIVE ]" if is_rss_active else "[ DRIVE ]"
    badge_font = fonts["badge"]
    badge_text_w = draw.textlength(mode_text, font=badge_font)
    badge_pad_x, badge_pad_y = 14, 8
    badge_x1 = w - 24
    badge_x0 = badge_x1 - (badge_text_w + badge_pad_x * 2)
    badge_y0, badge_y1 = 22, 22 + 18 + badge_pad_y * 2
    draw.rounded_rectangle(
        (badge_x0, badge_y0, badge_x1, badge_y1),
        radius=(badge_y1 - badge_y0) / 2,
        fill=(*accent, 235),
    )
    draw.text(
        (badge_x0 + badge_pad_x, badge_y0 + badge_pad_y - 1),
        mode_text,
        font=badge_font,
        fill=(255, 255, 255, 255),
    )

    # ================================================================
    # (2) 下部：3枚の情報カード（速度・制御／AIプランナー／RSS安全）
    # ================================================================
    card_h = 96
    card_gap = 16
    card_margin = 24
    card_w = (w - card_margin * 2 - card_gap * 2) / 3
    card_y0 = h - card_h - 24
    card_y1 = card_y0 + card_h

    # --- カード①：速度・制御 ---
    card1_x0 = card_margin
    card1_x1 = card1_x0 + card_w
    draw_card(
        draw,
        (card1_x0, card_y0, card1_x1, card_y1),
        title="Speed & Control",
        rows=[
            ("Speed", f"{telemetry['speed_kmh']:.1f} km/h"),
            ("Target", f"{telemetry['target_speed_kmh']:.1f} km/h"),
        ],
        fonts=fonts,
    )

    # --- カード②：AIプランナー ---
    card2_x0 = card1_x1 + card_gap
    card2_x1 = card2_x0 + card_w
    # 提案行動(AI Plan)と最終実行行動(Action)が異なる＝RSSが上書きした瞬間
    plan_overridden = telemetry["ai_plan"] != telemetry["final_action"]
    draw_card(
        draw,
        (card2_x0, card_y0, card2_x1, card_y1),
        title="AI Planner",
        rows=[
            ("AI Plan", telemetry["ai_plan"]),
            ("Action", telemetry["final_action"]),
        ],
        fonts=fonts,
        highlight=plan_overridden,
        accent_color=COLOR_ACCENT_ALERT,
    )

    # --- カード③：RSS安全メトリクス ---
    card3_x0 = card2_x1 + card_gap
    card3_x1 = card3_x0 + card_w
    gap_str = f"{telemetry['gap_m']:.1f} m" if telemetry["gap_m"] is not None else "—"
    margin_str = f"{telemetry['margin_m']:.1f} m" if telemetry["margin_m"] is not None else "—"
    draw_card(
        draw,
        (card3_x0, card_y0, card3_x1, card_y1),
        title="RSS Safety",
        rows=[
            ("Gap", gap_str),
            ("Margin", margin_str),
        ],
        fonts=fonts,
        highlight=is_rss_active,
        accent_color=COLOR_ACCENT_ALERT,
        footer_tag="[ OVERRIDE ]" if is_rss_active else None,
    )

    # --------------------------------------------------------------
    # 透明レイヤー(overlay)を元の走行シーン(base_img)に合成する
    # --------------------------------------------------------------
    composed = Image.alpha_composite(base_img, overlay).convert("RGB")
    return np.array(composed)


def run_visualization() -> None:
    """
    highway-env上でTrajectoryTransformer + RSSCheckerの推論パイプラインを
    実行しながら、Tesla FSD風UIを重ねた映像を1エピソード分録画し、
    config.VIDEO_OUTPUT_PATHへmp4として保存するメイン関数。
    """
    set_global_seed(config.VIS_SEED)

    env = gym.make("highway-v0", render_mode="rgb_array")
    # 【重要】collect_demo.pyでの学習データ収集時と全く同じ環境設定
    # （lanes_count / vehicles_density を含む）を使うこと。
    # ここがズレると、学習時とは異なる交通密度・車線数で評価することになり、
    # 「モデルの実力」ではなく「訓練/推論間の分布のズレ」を見てしまう
    # （train-test distribution mismatchの典型的な事故）。
    env.unwrapped.configure({
        "observation": {"type": "Kinematics"},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": config.TARGET_SPEEDS_MPS,
        },
        "vehicles_count": config.VEHICLES_COUNT,
        "lanes_count": config.LANES_COUNT,
        "vehicles_density": config.VEHICLES_DENSITY,
        "duration": config.VIS_NUM_STEPS,
        "policy_frequency": 5,
        # --- 描画まわりの設定（Tesla風のワイドなダッシュボード画角に寄せる） ---
        "screen_width": config.RENDER_WIDTH,
        "screen_height": config.RENDER_HEIGHT,
        "scaling": config.RENDER_SCALING,
    })

    feature_dim = int(np.prod(env.observation_space.shape))
    print(f"[visualize] 観測の形状: {env.observation_space.shape} "
          f"→ feature_dim = {feature_dim}")

    model = TrajectoryTransformer(
        feature_dim=feature_dim,
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        num_actions=NUM_ACTIONS,
        max_seq_len=config.SEQ_LEN,
    )
    load_model_weights_if_available(model)  # train_irl.py の学習済み重みがあれば使う

    rss_checker = RSSChecker(RSSParameters())
    fonts = load_fonts()

    # --------------------------------------------------------------
    # 動画ライターの準備。
    # 描画解像度(config.RENDER_WIDTH x config.RENDER_HEIGHT)と
    # 完全に一致させる必要がある点に注意（サイズが違うと書き込みに失敗する）。
    # --------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        config.VIDEO_OUTPUT_PATH,
        fourcc,
        config.VIDEO_FPS,
        (config.RENDER_WIDTH, config.RENDER_HEIGHT),
    )
    if not video_writer.isOpened():
        raise RuntimeError(
            "VideoWriterのオープンに失敗しました。"
            "OpenCVがmp4vコーデックに対応しているか確認してください。"
        )

    obs, info = env.reset(seed=config.VIS_SEED)
    history = deque(maxlen=config.SEQ_LEN)
    pressure_streak = 0                    # SLOWER提案 or RSS介入が連続した回数
    lane_change_lock_direction = None      # 進行中の車線変更方向(-1:左 / +1:右)
    lane_change_lock_steps_left = 0        # ロックの残りステップ数
    lane_change_lock_target_lane = None    # ロック開始時に狙った車線ID

    print(f"[visualize] 録画開始（seed={config.VIS_SEED}, "
          f"最大{config.VIS_NUM_STEPS}ステップ, "
          f"出力先={config.VIDEO_OUTPUT_PATH}）")

    frame_count = 0
    # 録画全体を通じた行動分布（クラス不均衡対策の効果を動画とあわせて
    # 数値でも確認できるようにするための集計）
    action_counter = new_action_counter()

    for step in range(config.VIS_NUM_STEPS):
        # --- (a) 上位プランナーが行動を提案する（main.pyと全く同じ処理） ---
        obs_flat = flatten_observation(obs)
        history.append(obs_flat)
        proposed_action, reward_score = propose_action_by_model(
            model, history, config.SEQ_LEN, feature_dim
        )

        ego = env.unwrapped.vehicle

        # --- (b) 車線変更のラッチ（main.pyと同じロジック） ---
        if lane_change_lock_direction is not None:
            lane_change_lock_steps_left -= 1
            opposite_action = (
                ACTION_LANE_RIGHT if lane_change_lock_direction == -1
                else ACTION_LANE_LEFT
            )
            if proposed_action == opposite_action:
                proposed_action = ACTION_IDLE
            if lane_change_lock_steps_left <= 0:
                lane_change_lock_direction = None

        # --- (c) RSS用の情報取得（main.pyと全く同じ処理） ---
        front_info = get_front_vehicle_info(env)
        if front_info is not None:
            distance, v_rear, v_front = front_info
            margin = rss_checker.compute_min_safe_distance(v_rear, v_front)
            is_rss_active = not rss_checker.is_safe(distance, v_rear, v_front)
            was_safe = not is_rss_active
            
        else:
            distance, margin, is_rss_active, was_safe = None, None, False, True
            v_rear, v_front = None, None

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

        # --- (d) 早期Nudge / 緊急回避の判定（main.pyと同じロジック） ---
        danger_signal = not was_safe
        pressure_streak = pressure_streak + 1 if danger_signal else 0

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

        # --- (e) RSSは最終防衛ラインとして必ず安全判定を行う（スキップしない）。
        # lane_change_candidateがNoneなら、従来と全く同じ「危険なら必ず
        # ブレーキ」の動作になる ---
        if front_info is not None:
            final_action = rss_checker.override_action(
                current_distance=distance,
                v_rear=v_rear,
                v_front=v_front,
                proposed_action=effective_proposed_action,
                brake_action=4,  # SLOWER
                lane_change_action=lane_change_candidate,
            )
        else:
            final_action = effective_proposed_action

        # --- (c) AIプランナーの提案行動を集計する（動画とは別に、数値ログとしても残す） ---
        action_counter[ACTION_NAMES[proposed_action]] += 1

        # --- (d) 現在の車両状態を取得し、UI表示用のテレメトリを組み立てる ---
        ego = env.unwrapped.vehicle
        telemetry = {
            "speed_kmh": ego.speed * 3.6,
            "target_speed_kmh": ego.target_speed * 3.6,
            "ai_plan": ACTION_NAMES[proposed_action],
            "final_action": ACTION_NAMES[final_action],
            "gap_m": distance,
            "margin_m": margin,
            "rss_active": is_rss_active,
            "irl_reward": reward_score,
        }

        # --- (e) 走行シーンを描画し、UIレイヤーを重ねて1フレーム分完成させる ---
        frame_rgb = env.render()
        frame_with_ui = render_ui_overlay(frame_rgb, telemetry, fonts)

        # PILはRGB、OpenCVの動画書き出しはBGRを前提としているため変換する
        frame_bgr = cv2.cvtColor(frame_with_ui, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        frame_count += 1

        # --- (f) 環境を1ステップ進める ---
        obs, reward, terminated, truncated, info = env.step(final_action)

        if terminated or truncated:
            print(f"  → ステップ{step}でエピソード終了 "
                  f"(terminated={terminated}, truncated={truncated})")
            break

    video_writer.release()
    env.close()

    print(f"[visualize] 録画完了：{frame_count}フレームを "
          f"{config.VIDEO_OUTPUT_PATH} に保存しました "
          f"（{config.VIDEO_FPS}fps ≒ 約{frame_count / config.VIDEO_FPS:.1f}秒）。")

    # ------------------------------------------------------------
    # 録画中にAIプランナーが提案した行動の分布を表示する。
    # 「動画を見た印象」だけでなく、この数値でも
    # 「IDLE/SLOWERへの偏りが解消されたか」「LANE_LEFT/RIGHTを
    #  必要な場面で選べているか」を確認できるようにしている。
    # ------------------------------------------------------------
    print()
    print(format_action_distribution(action_counter, "録画エピソード中の行動分布"))


if __name__ == "__main__":
    run_visualization()