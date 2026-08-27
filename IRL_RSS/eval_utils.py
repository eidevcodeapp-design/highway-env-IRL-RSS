"""
eval_utils.py
=====================================================================
【概要】
main.py（推論評価）と visualize.py（可視化）の両方で共通して使う、
「AIプランナー（TrajectoryTransformer）が提案した行動の出現分布」を
集計・整形して表示するための小さなユーティリティ関数群。

【なぜこのファイルが必要か】
第3段階までは「動画を目で見て、なんとなく保守的に見える」という
定性的な確認しかできていなかった。
第4段階（クラス不均衡対策）の効果を検証するには、
    「LANE_LEFT / LANE_RIGHT を選ぶ割合が実際に増えたか」
を"数値で"示す必要がある。
main.py と visualize.py の両方に同じ集計ロジックをコピペすると、
将来どちらか一方だけ直して数値の定義がズレる事故が起きやすいため、
共通処理としてこのファイルに切り出している。
=====================================================================
"""

from transformer_planner import ACTION_NAMES


def new_action_counter() -> dict:
    """
    行動名(ACTION_NAMES)をキーとする、件数0で初期化されたカウンタ辞書を作る。

    Returns
    -------
    dict
        例: {"LANE_LEFT": 0, "IDLE": 0, "LANE_RIGHT": 0, "FASTER": 0, "SLOWER": 0}
    """
    return {name: 0 for name in ACTION_NAMES}


def merge_action_counters(base: dict, other: dict) -> None:
    """
    otherのカウントをbaseに加算する（in-place）。
    複数エピソード分の内訳を、実行全体の集計に積み上げる際に使う。
    """
    for name, count in other.items():
        base[name] = base.get(name, 0) + count


def format_action_distribution(action_counter: dict, title: str) -> str:
    """
    行動カウンタを「件数・比率(%)」つきの文字列に整形する。

    LANE_LEFT + LANE_RIGHT の合計比率も併記することで、
    「保守的なIDLE/SLOWERに偏っていないか」「クラス不均衡対策の効果が
    出ているか」が一目で分かるようにしている。
    第3段階で確認した"約4%"という数値と、再学習後の数値を見比べることで、
    改善効果をそのままポートフォリオの根拠として使える。

    Parameters
    ----------
    action_counter : dict
        new_action_counter() で作成し、行動が選ばれるたびに
        インクリメントしてきたカウンタ辞書
    title : str
        見出し文字列（例: "エピソード1の行動分布", "全エピソード合計"）

    Returns
    -------
    str
        そのままprint()に渡せる、整形済みの複数行文字列
    """
    total = sum(action_counter.values())
    lines = [f"  ▼ {title} (総サンプル数={total}):"]
    for name in ACTION_NAMES:
        count = action_counter.get(name, 0)
        ratio = 100.0 * count / max(1, total)
        bar = "█" * int(ratio // 4)  # 簡易テキストバー（4%刻み）
        lines.append(f"      {name:>10s}: {count:5d}回 ({ratio:5.1f}%) {bar}")

    lane_change = action_counter.get("LANE_LEFT", 0) + action_counter.get("LANE_RIGHT", 0)
    lane_change_ratio = 100.0 * lane_change / max(1, total)
    lines.append(
        f"      ---- 車線変更(LANE_LEFT+LANE_RIGHT)合計: "
        f"{lane_change:5d}回 ({lane_change_ratio:5.1f}%) ----"
    )
    return "\n".join(lines)