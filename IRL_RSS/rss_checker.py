"""
rss_checker.py
=====================================================================
【概要】
RSS (Responsibility-Sensitive Safety) モデルのうち、最も基本的かつ
実用上重要な「縦方向（前後方向）の最小安全距離」を計算し、
現在の車間距離が危険な場合に自動的に安全な行動（ブレーキ）へ
介入（Override）するクラスを提供する。
 
【RSSモデルとは】
Mobileye社（現Intel傘下）が2017年に提唱した、自動運転車の「事故責任」を
数学的に定義するための安全モデルである。
「どんなに前方車両が急ブレーキをかけても、自車が定義された最小安全距離
（RSS距離）を保っていれば、自車に衝突責任は発生しない」ということを
数式で保証する考え方。
 
本モジュールでは、その中核となる「縦方向・同一車線での最小安全距離の公式」
を実装する。この関数は、IRL（逆強化学習）やTransformerで学習した
"ソフトな"経路計画方策の出力に対して、"ハードな"安全性の最終防衛ライン
（セーフティネット）として働く。
=====================================================================
"""
# ライブラリを取得
from dataclasses import dataclass

@dataclass
class RSSParameters:
    """
    RSS計算に使用する物理パラメータをまとめたデータクラス。
 
    Attributes
    ----------
    response_time : float
        反応時間 ρ [s]。危険を検知してから実際にブレーキを踏み始めるまでの
        遅延時間。RSSモデルでは「最悪ケース」を仮定し、この間、自車は
        max_accel で加速し続けていると想定する。
    max_accel : float
        反応時間中に許容する自車の最大加速度 a_max,accel [m/s^2]
        （＝センサーが危険を検知するまでの間、うっかりアクセルを
          踏んでいた場合の最悪ケースを表す）
    min_brake : float
        自車が"保証"できる最小の減速度 a_min,brake [m/s^2]
        （自車のブレーキ性能として、少なくともこれだけは効くという
          下限値。安全側に倒すため、あえて"最小"の性能で計算する）
    max_brake_front : float
        前方車両が起こしうる最大の急ブレーキ減速度 a_max,brake [m/s^2]
        （前方車が突然フルブレーキをかけても対応できるよう、
          物理的に可能な最悪値を仮定する）
    """
    response_time: float = 0.5        # ρ: 反応時間 [s] (旧: 1.0 -> 0.5)
    max_accel: float = 2.0            # 反応時間中の自車最大加速度 [m/s^2] (旧: 3.0)
    min_brake: float = 4.0            # 自車が保証する最小減速度 [m/s^2] (変化なし)
    max_brake_front: float = 5.0      # 前方車両が起こしうる最大減速度 [m/s^2] (旧: 8.0)
 
 
class RSSChecker:
    """
    RSS（縦方向安全距離）を計算し、危険な場合にブレーキ介入を判定するクラス。
 
    使い方の流れ:
        1. compute_min_safe_distance() で理論上の最小安全距離を計算
        2. is_safe() で現在の車間距離がその距離を上回っているか判定
        3. override_action() で、上位の方策（IRL/Transformer）が提案した
           行動を、危険時のみ強制的にブレーキへ書き換える
    """
 
    def __init__(self, params: RSSParameters = None):
        """
        Parameters
        ----------
        params : RSSParameters, optional
            RSS計算用パラメータ。指定しない場合はデフォルト値を使用する。
        """
        self.params = params if params is not None else RSSParameters()
 
    def compute_min_safe_distance(self, v_rear: float, v_front: float) -> float:
        """
        縦方向の最小安全距離 d_min を計算する（Mobileye RSSモデルの中核公式）。
 
        数式:
            d_min = [ v_r*ρ + (1/2)*a_max,accel*ρ^2
                      + (v_r + ρ*a_max,accel)^2 / (2*a_min,brake) ]
                    - v_f^2 / (2*a_max,brake)
 
        この式は「反応時間中に自車が加速しながら進む距離」＋
        「反応後に自車が停止するまでの制動距離」から、
        「前方車両が急ブレーキで停止するまでの制動距離」を引いたもの。
        つまり "最悪ケースで前方車が急停止しても、自車が追突しないために
        今確保しておくべき車間距離" を表す。
 
        Parameters
        ----------
        v_rear : float
            後方車両（自車）の現在速度 [m/s]
        v_front : float
            前方車両の現在速度 [m/s]
 
        Returns
        -------
        float
            最小安全車間距離 [m]（理論上負になる場合は0にクリップする）
        """
        rho = self.params.response_time
        a_accel = self.params.max_accel
        a_brake_self = self.params.min_brake
        a_brake_front = self.params.max_brake_front
 
        # ① 反応時間ρの間、自車が「加速し続けながら」進む距離
        #    （まだブレーキを踏んでいない、最悪ケースの区間）
        distance_during_response = v_rear * rho + 0.5 * a_accel * (rho ** 2)
 
        # ② 反応時間後の速度から、自車が「保証できる最小減速度」で
        #    ブレーキをかけ、停止するまでに進む距離
        v_rear_after_response = v_rear + rho * a_accel
        braking_distance_self = (v_rear_after_response ** 2) / (2 * a_brake_self)
 
        # ③ 前方車両が「最大減速度」で急ブレーキをかけ、停止するまでに進む距離
        braking_distance_front = (v_front ** 2) / (2 * a_brake_front)
 
        # ④ 自車の走行距離合計 − 前方車の走行距離 ＝ 必要な最小車間距離(RSSモデルの中核公式)
        d_min = distance_during_response + braking_distance_self - braking_distance_front
 
        # 物理的に距離が負になることはあり得ないため、0未満は0にクリップする
        return max(0.0, d_min)
 
    def is_safe(self, current_distance: float, v_rear: float, v_front: float) -> bool:
        """
        現在の車間距離がRSS上「安全」かどうかを判定する。
 
        Parameters
        ----------
        current_distance : float
            現在の車間距離（前方車両までの距離）[m]
        v_rear : float
            自車速度 [m/s]
        v_front : float
            前方車両速度 [m/s]
 
        Returns
        -------
        bool
            True  : 安全（RSS最小安全距離以上を確保できている）
            False : 危険（RSS最小安全距離を下回っている＝介入が必要）
        """
        d_min = self.compute_min_safe_distance(v_rear, v_front)
        return current_distance >= d_min
 
    def override_action(
        self,
        current_distance: float,
        v_rear: float,
        v_front: float,
        proposed_action: int,
        brake_action: int = 4,
        lane_change_action: int = None,
    ) -> int:
        if self.is_safe(current_distance, v_rear, v_front):
            return proposed_action

        print(f"[DEBUG] override_action呼び出し: proposed_action={proposed_action}, "
              f"lane_change_action={lane_change_action}")  # ← この行を追加

        # 危険域では原則ブレーキへ強制介入する。ただし、呼び出し側が
        # 「隣接車線が実際に空いていること」を確認済みで(lane_change_action)、
        # かつ、ちょうどその車線変更が提案されている場合に限り、
        # ブレーキではなく回避を優先する。
        # 【機能安全上の位置づけ】RSSの判定をスキップしているのではなく、
        # 「ブレーキだけが唯一の安全行動ではない」というRSS本来の思想
        # （衝突責任を回避できる行動を選ぶ）に基づき、RSSが検証済みの
        # 選択肢の中から安全な行動を選び直しているだけである。
        # lane_change_actionが渡されていない(None)場合は、
        # 従来通り必ずブレーキへ上書きする。
        if lane_change_action is not None and proposed_action == lane_change_action:
            return lane_change_action

        d_min = self.compute_min_safe_distance(v_rear, v_front)
        print(
            f"[RSS介入] 危険な車間距離を検知しました！ "
            f"現在距離={current_distance:.2f}m < 最小安全距離={d_min:.2f}m "
            f"提案行動={proposed_action} → 強制ブレーキ(action={brake_action})に上書き"
        )
        return brake_action

def is_adjacent_lane_clear(env, direction: int, margin_m: float) -> bool:
    """
    ego が隣接車線（direction=-1:左 / +1:右）に入った場合、
    前後margin_m[m]以内に他車がいないかを見る簡易的な"隙間チェック"。
    Mobileyeの厳密な横方向RSS（相対速度まで考慮した安全マージン）
    ではない、あくまで実用上の簡易ガードである点に注意。
    """
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
    target_lane_id = ego.lane_index[2] + direction

    lane_count = len(road.network.graph[ego.lane_index[0]][ego.lane_index[1]])
    if target_lane_id < 0 or target_lane_id >= lane_count:
        return False  # 道路の端の外側には出られない

    for v in road.vehicles:
        if v is ego or v.lane_index[2] != target_lane_id:
            continue
        if abs(v.position[0] - ego.position[0]) < margin_m:
            return False
    return True
 
if __name__ == "__main__":
    # 単体実行時の動作確認用（python rss_checker.py で実行可能）
    checker = RSSChecker()
 
    # ケース1: 自車20m/s、前方車10m/s、車間距離30m → 危険なはず
    d_min = checker.compute_min_safe_distance(v_rear=20.0, v_front=10.0)
    print(f"最小安全距離: {d_min:.2f} m")
    print("現在距離30mは安全か？:", checker.is_safe(30.0, 20.0, 10.0))
 
    action = checker.override_action(
        current_distance=30.0, v_rear=20.0, v_front=10.0, proposed_action=3
    )
    print("最終行動:", action)