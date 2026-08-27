"""
transformer_planner.py
=====================================================================
【概要】
本モジュールは、過去Tステップ分の「自車・周辺車両の相対位置／速度」の
時系列データを入力とし、TransformerEncoderで時系列特徴を集約した上で、
以下の2つの出力を行う `TrajectoryTransformer` クラスを提供する。
 
    (A) 方策ヘッド (Policy Head)
        「次にどの行動を取るべきか」の5行動分のロジット（未正規化スコア）
        を出力する。これが上位プランナーとして main.py から呼び出される。
 
    (B) IRLヘッド / 報酬ヘッド (Reward Head)
        逆強化学習（IRL: Inverse Reinforcement Learning）の考え方に基づき、
        「ある状態である行動を取ることが、どれだけ"人間の熟練ドライバー"
        らしいか」を表すスカラー値（Reward Score）を出力する。
 
【IRLとTransformer方策の関係（重要な考え方）】
    通常の強化学習(RL)では「人間が設計した報酬関数」を最大化するように
    方策を学習するが、"安全かつ自然な運転"のような複雑な振る舞いを
    人間が数式で完璧に定義するのは非常に困難です。
 
    逆強化学習(IRL)は逆に、「熟練ドライバーの運転データ（デモンストレーション）」
    から「その人がどんな報酬関数を暗黙的に最大化していたか」を推定する。
    本クラスの reward_head はこの"推定された報酬関数"の役割を担い、
    policy_head はその報酬のもとで「最適に近い」行動を出力する方策の
    役割を担う。
 
    このファイル単体では、まだIRLの学習（報酬関数を人間データに
    フィッティングさせる処理）は実装していない。
    今回はまず「推論時にどう使うか」というアーキテクチャの土台を
    作ることが目的である。学習ループ（デモデータ収集 → 報酬関数の最適化）は
    次のステップで実装する。
=====================================================================
"""

# 必要なライブラリを取得
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 行動の定義（main.py / rss_checker.py と一貫性を持たせるための定数）
# =====================================================================
NUM_ACTIONS = 5  # highway-envのDiscreteMetaActionに対応する行動の数
ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
 
 
def to_action_onehot(action_id: int, num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """
    行動ID（整数）を one-hot ベクトルに変換するヘルパー関数。
 
    IRLの報酬ヘッドは「状態」と「行動」の両方を入力として受け取る必要が
    あるため、行動を数値ベクトルとして表現する必要がある。
    ここでは最もシンプルな one-hot 表現を採用する。
 
    Parameters
    ----------
    action_id : int
        行動ID（0〜num_actions-1）
    num_actions : int, default=NUM_ACTIONS
        行動の総数
 
    Returns
    -------
    torch.Tensor
        shape = [num_actions] の one-hot ベクトル
    """
    onehot = torch.zeros(num_actions, dtype=torch.float32)
    onehot[action_id] = 1.0
    return onehot
 
 
class PositionalEncoding(nn.Module):
    """
    Transformerに「時刻の順序情報」を与えるための位置エンコーディング層。
 
    【なぜ必要か】
    TransformerのSelf-Attention機構は、本質的には「集合（順序を持たない
    データの集まり）」に対する処理であり、それ自体には「どのステップが
    過去でどのステップが現在に近いか」という時間の前後関係の情報が
    含まれていない。
 
    そのため、入力の特徴ベクトルに対して、時刻ごとに異なるパターンを持つ
    サイン・コサイン波（位置エンコーディング）を"加算"することで、
    モデルが時間的な順序を区別できるようにする。
    （これはOriginal Transformer論文 "Attention Is All You Need" と
      同じ標準的な実装方式）
    """
 
    def __init__(self, d_model: int, max_len: int = 500):
        """
        Parameters
        ----------
        d_model : int
            Transformer内部の特徴次元数（埋め込み次元）
        max_len : int, default=500
            想定する最大の時系列長（この長さ分の位置エンコーディングを
            事前に計算してテーブルとして保持しておく）
        """
        super().__init__()
 
        # 位置エンコーディングのテーブルを事前計算しておく
        # shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
 
        # 各時刻position（0, 1, 2, ...）を列ベクトルとして用意
        # shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
 
        # サイン・コサインの周波数を決める分母項（次元ごとに周波数を変える）
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
 
        # 偶数次元にはsin、奇数次元にはcosを割り当てる（標準的な実装）
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        # バッチ次元を追加してshapeを [1, max_len, d_model] にしておく
        # → forward時にブロードキャストでバッチ全体に加算できるようにするため
        pe = pe.unsqueeze(0)
 
        # 学習対象パラメータではないが、モデルの一部として保存したいので
        # register_buffer を使う（state_dict()には含まれるが勾配計算はされない）
        self.register_buffer("pe", pe)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルに位置エンコーディングを加算する。
 
        Parameters
        ----------
        x : torch.Tensor
            shape = [batch_size, seq_len, d_model]
 
        Returns
        -------
        torch.Tensor
            shape = [batch_size, seq_len, d_model]（入力と同じ形状）
            各時刻の特徴ベクトルに、対応する位置エンコーディングが加算されている
        """
        seq_len = x.size(1)
        # self.pe は [1, max_len, d_model] なので、実際の系列長分だけ切り出して加算する
        x = x + self.pe[:, :seq_len, :]
        return x
 
 
class TrajectoryTransformer(nn.Module):
    """
    時系列の運転状況（自車・周辺車両の相対位置／速度）を入力として、
    (A) 次の行動のロジット（方策ヘッド）
    (B) 状態-行動対の"人間らしさ"報酬スコア（IRL報酬ヘッド）
    の2つを出力するTransformerベースのモデル。
 
    【全体のテンソルの流れ（重要）】
 
        入力 obs_history
        shape: [batch_size, seq_len, feature_dim]
              │
              ▼ (1) input_proj: 特徴量次元を圧縮/拡張してTransformerの内部次元に合わせる
        shape: [batch_size, seq_len, d_model]
              │
              ▼ (2) pos_encoder: 各時刻に位置エンコーディングを加算
        shape: [batch_size, seq_len, d_model]  （形状は変化しない）
              │
              ▼ (3) transformer_encoder: Self-Attentionで時系列全体の関係を集約
        shape: [batch_size, seq_len, d_model]  （形状は変化しない）
              │
              ▼ (4) 最後の時刻（＝現在に最も近いステップ）のベクトルだけを取り出す
        shape: [batch_size, d_model]
              │
              ├─▶ (5a) policy_head: 行動ロジットを出力
              │  shape: [batch_size, num_actions]
              │
              └─▶ (5b) reward_head: 状態ベクトル+行動one-hotを結合して報酬を出力
                 shape: [batch_size, 1]
 
    このように「最後の時刻のベクトル」を"現在の状態表現"として扱うのは、
    Self-Attentionによって既に過去T ステップ分の情報がその1本のベクトルに
    "集約"されているためである（RNNの最終隠れ状態を使う考え方に近い）。
    """
 
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_actions: int = NUM_ACTIONS,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        feature_dim : int
            1ステップあたりの生の観測特徴量の次元数。
            例）highway-envのKinematics観測で「周辺車両5台 × 5特徴量
            (presence, x, y, vx, vy)」の場合、flatten後は 5*5=25 になる。
        d_model : int, default=64
            Transformer内部で扱う特徴ベクトルの次元数（埋め込み次元）。
            大きいほど表現力は上がるが、計算コストとメモリ使用量も増える。
        nhead : int, default=4
            Multi-Head Attentionのヘッド数。d_modelはnheadで割り切れる
            必要がある（64 / 4 = 16、1ヘッドあたり16次元を担当する）。
        num_layers : int, default=2
            TransformerEncoderLayerを何層重ねるか。
        dim_feedforward : int, default=128
            各Encoder層内のFeed-Forwardネットワークの中間層次元数。
        num_actions : int, default=NUM_ACTIONS(=5)
            出力する行動の種類数。
        max_seq_len : int, default=50
            位置エンコーディングが対応できる最大時系列長。
        dropout : float, default=0.1
            過学習を防ぐためのドロップアウト率。
        """
        super().__init__()
 
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_actions = num_actions
 
        # ---------------------------------------------------------------
        # (1) 入力射影層：生の観測特徴量(feature_dim次元)をTransformerの
        #     内部次元(d_model次元)へ線形変換する。
        #     例：25次元の観測 → 64次元の内部表現へ拡張
        # ---------------------------------------------------------------
        self.input_proj = nn.Linear(feature_dim, d_model)
 
        # ---------------------------------------------------------------
        # (2) 位置エンコーディング：時系列の順序情報を付与する
        # ---------------------------------------------------------------
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
 
        # ---------------------------------------------------------------
        # (3) TransformerEncoder本体
        #     batch_first=True にすることで、入出力テンソルの形状を
        #     [batch_size, seq_len, d_model] という直感的な順序に統一できる
        #     （Transformer登場当初のデフォルトは [seq_len, batch, dim] だった点に注意）
        # ---------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
 
        # ---------------------------------------------------------------
        # (5a) 方策ヘッド（Policy Head）
        #      現在の状態表現(d_model次元) → 各行動のロジット(num_actions次元)
        #      ロジットはまだ確率化される前の"生スコア"であり、
        #      softmaxを通すことで行動確率分布に変換できる。
        # ---------------------------------------------------------------
        self.policy_head = nn.Linear(d_model, num_actions)
 
        # ---------------------------------------------------------------
        # (5b) IRL報酬ヘッド（Reward Head）
        #      「状態表現(d_model次元)」+「行動one-hot(num_actions次元)」
        #      を結合したベクトルを入力とし、スカラーの報酬値を出力する
        #      2層のMLP（多層パーセプトロン）。
        #      state-action pairから報酬を出す設計は、逆強化学習でよく使われる
        #      "Reward Network" の典型的な構成。
        # ---------------------------------------------------------------
        self.reward_head = nn.Sequential(
            nn.Linear(d_model + num_actions, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
 
    def forward(self, obs_history: torch.Tensor):
        """
        時系列観測データを入力として、行動ロジットと状態表現を出力する。
 
        Parameters
        ----------
        obs_history : torch.Tensor
            shape = [batch_size, seq_len, feature_dim]
            過去 seq_len ステップ分の観測を時系列順（古い→新しい）に
            並べたテンソル。
 
        Returns
        -------
        action_logits : torch.Tensor
            shape = [batch_size, num_actions]
            各行動の未正規化スコア（そのままargmaxを取れば決定的な行動選択、
            softmaxを通せば確率分布になる）
        state_repr : torch.Tensor
            shape = [batch_size, d_model]
            現在の状態を要約したベクトル表現。IRL報酬ヘッドの入力としても使う。
        """
        # --- (1) 入力射影：feature_dim → d_model ---
        # x: [batch_size, seq_len, feature_dim] → [batch_size, seq_len, d_model]
        x = self.input_proj(obs_history)
 
        # --- (2) 位置エンコーディングを加算 ---
        # 形状は変化しない: [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
 
        # --- (3) TransformerEncoderでSelf-Attentionを適用 ---
        # 各時刻のベクトルが、系列内の"他の全ての時刻"の情報を加味して
        # 更新される（＝過去の履歴全体を考慮した表現になる）。
        # 形状は変化しない: [batch_size, seq_len, d_model]
        #
        # 注：本モデルは「未来を予測する」のではなく「既知の過去履歴から
        # "現在"の状態を要約する」ことが目的のため、GPTのような
        # Causal Mask（未来の情報を見せないためのマスク）は使用していない。
        # 系列内の全ステップは既に観測済みの"過去"データだからである。
        encoded = self.transformer_encoder(x)
 
        # --- (4) 最後の時刻（=最新の観測）のベクトルを"現在の状態表現"として取り出す ---
        # encoded: [batch_size, seq_len, d_model]
        #        → state_repr: [batch_size, d_model]
        # (Self-Attentionにより、この1本のベクトルには過去seq_len分の
        #  情報が既に"集約"されている点がポイント)
        state_repr = encoded[:, -1, :]
 
        # --- (5a) 方策ヘッドで行動ロジットを計算 ---
        # state_repr: [batch_size, d_model] → action_logits: [batch_size, num_actions]
        action_logits = self.policy_head(state_repr)
 
        return action_logits, state_repr
 
    def compute_reward(
        self, state_repr: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        IRL報酬ヘッドを使い、「状態-行動対」に対する"人間らしさ"の
        報酬スコアを計算する。
 
        Parameters
        ----------
        state_repr : torch.Tensor
            shape = [batch_size, d_model]
            forward()で得られた状態表現ベクトル。
        action_onehot : torch.Tensor
            shape = [batch_size, num_actions]
            評価したい行動のone-hotベクトル
            （to_action_onehot()で作成したものをバッチ化して渡す）。
 
        Returns
        -------
        torch.Tensor
            shape = [batch_size, 1]
            "人間の熟練ドライバーらしさ"を表すスカラー報酬スコア。
            値そのものに単位はなく、あくまで「他の行動候補と比較して
            相対的に高いか低いか」に意味がある（IRLの一般的な性質）。
            現時点ではモデルは未学習のため、この値はまだランダムに近い。
            将来、人間の運転デモンストレーションデータを使って
            この報酬ヘッドを学習させることで、意味のあるスコアになる。
        """
        # 状態ベクトルと行動one-hotベクトルを、特徴次元方向(dim=-1)に連結する
        # state_repr: [batch_size, d_model]
        # action_onehot: [batch_size, num_actions]
        # combined: [batch_size, d_model + num_actions]
        combined = torch.cat([state_repr, action_onehot], dim=-1)
 
        # 2層MLPを通してスカラー報酬を出力
        # combined: [batch_size, d_model+num_actions] → reward: [batch_size, 1]
        reward = self.reward_head(combined)
        return reward
    
    def compute_all_action_rewards(self, state_repr: torch.Tensor) -> torch.Tensor:
        """train_irl.pyのcompute_rewards_for_all_actions()と同じ計算を、
        推論側(main.py)でも使えるようモデルのメソッドとして切り出したもの。"""
        batch_size = state_repr.size(0)
        device = state_repr.device
        reward_list = []
        for action_id in range(NUM_ACTIONS):
            onehot = torch.zeros(batch_size, NUM_ACTIONS, device=device)
            onehot[:, action_id] = 1.0
            reward_list.append(self.compute_reward(state_repr, onehot))
        return torch.cat(reward_list, dim=1)  # [batch_size, num_actions]
 
    def select_action(
        self, action_logits: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """
        方策ヘッドが出力したロジットから、実際に実行する行動IDを選択する。
 
        Parameters
        ----------
        action_logits : torch.Tensor
            shape = [batch_size, num_actions]
        deterministic : bool, default=True
            True  : 最もスコアの高い行動を常に選ぶ（argmax、決定的＝再現性が高い）
            False : softmaxで確率化した分布からサンプリングする
                    （探索的な行動選択。学習時の"方策のばらつき"を作りたい場合に使用）
 
        Returns
        -------
        torch.Tensor
            shape = [batch_size]
            選択された行動ID（整数）のテンソル
        """
        if deterministic:
            # 各バッチについて、num_actions次元の中で最大値を取るインデックスを選ぶ
            # action_logits: [batch_size, num_actions] → action: [batch_size]
            action = torch.argmax(action_logits, dim=-1)
        else:
            # ロジットを確率分布に変換してからサンプリングする
            probs = F.softmax(action_logits, dim=-1)  # [batch_size, num_actions]
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]
        return action
 
 
if __name__ == "__main__":
    # 単体実行時の動作確認用（python transformer_planner.py で実行可能）
    # ダミーデータでテンソル形状が正しく流れるかを検証する。
    batch_size = 1
    seq_len = 10
    feature_dim = 25  # highway-env: 5台 × 5特徴量 を想定
 
    model = TrajectoryTransformer(feature_dim=feature_dim)
 
    dummy_obs_history = torch.randn(batch_size, seq_len, feature_dim)
    logits, state_repr = model(dummy_obs_history)
    print("action_logits shape:", logits.shape)   # 期待値: [1, 5]
    print("state_repr shape:", state_repr.shape)   # 期待値: [1, 64]
 
    action_id = model.select_action(logits, deterministic=True)
    print("選択された行動:", action_id.item(), ACTION_NAMES[action_id.item()])
 
    action_onehot = to_action_onehot(action_id.item()).unsqueeze(0)  # [1, num_actions]
    reward = model.compute_reward(state_repr, action_onehot)
    print("IRL報酬スコア:", reward.item())