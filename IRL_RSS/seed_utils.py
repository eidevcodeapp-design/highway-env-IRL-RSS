"""
seed_utils.py
=====================================================================
【概要】
このモジュールは、機械学習・強化学習の実験における「再現性
（Reproducibility）」を担保するために、Python標準のrandom、NumPy、
PyTorch のシードを一括で固定するユーティリティ関数を提供する。
 
【なぜ再現性が重要か】
自動運転AIの評価では「同じ条件で何度実験しても同じ結果になる」ことが
非常に重要である。シードが固定されていないと、実験のたびに
・他車の初期配置
・自車の初期速度
・ニューラルネットワークの重み初期化
などがランダムに変化してしまい、「アルゴリズムを改善したから性能が
上がったのか」「単なる乱数の運が良かっただけなのか」を区別できなく
なってしまう。
=====================================================================
"""

# 必要なライブラリをインポートする
import os
import random

import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """
    Python / NumPy / PyTorch のシードをまとめて固定する関数。
 
    Parameters
    ----------
    seed : int, default=42
        固定したい乱数シード値。
 
    Notes
    -----
    - Gymnasium / highway-env 自体の内部乱数（他車の初期配置や挙動など）
      は、この関数だけでは固定されない。
      必ず main.py 側で `env.reset(seed=seed)` のように、reset時に
      seedを明示的に渡すようにする。
    - PyTorchのCuDNN（GPU上の畳み込み演算高速化ライブラリ）は、デフォルトでは
      「非決定的（＝毎回わずかに計算順序が変わる）」アルゴリズムを使うことが
      あるため、ここで明示的に「決定的アルゴリズムのみを使う」よう設定する。
      （代わりに多少の速度低下とのトレードオフが発生する）
    """
    # ① Python標準のrandomモジュールのシードを固定
    #    → リストのシャッフルや random.choice() などに影響
    random.seed(seed)
 
    # ② NumPyのシードを固定
    #    → highway-env内部の乱数生成や、IRLでの報酬サンプリング等に影響
    np.random.seed(seed)
 
    # ③ 環境変数 PYTHONHASHSEED を固定
    #    → dict / set の反復順序など、ハッシュ値に依存する挙動の再現性を確保
    #    （ただし現在実行中のプロセスには反映されないため、
    #      本来は起動時の環境変数として設定するのが望ましい点に注意）
    os.environ["PYTHONHASHSEED"] = str(seed)
 
    # ④ PyTorchのCPU計算用シードを固定
    #    → Transformerモデルの重み初期化、Dropoutのマスク生成等に影響
    torch.manual_seed(seed)
 
    # ⑤ PyTorchのGPU(CUDA)用シードを固定
    #    GPUが無い環境でも安全に呼び出せるよう、存在チェックを行う
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       # 単一GPU用
        torch.cuda.manual_seed_all(seed)   # マルチGPU環境用
 
    # ⑥ CuDNNの決定的アルゴリズムを強制し、非決定的な高速化処理を無効化
    #    → 同じ入力に対して常に同じ計算順序・同じ結果を保証する
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
    print(f"[seed_utils] 乱数シードを {seed} に固定した。"
          f"(random / numpy / torch / cudnn)")
 
 
def get_seeded_worker_init_fn(seed: int = 42):
    """
    PyTorchのDataLoaderへ渡す worker_init_fn を生成するヘルパー関数（拡張用）。
 
    IRLの学習データをDataLoaderで並列読み込みする場合、複数のworker
    プロセスそれぞれで乱数がズレてしまうことがある。
    この関数を使うと、各workerに「ベースシード + worker番号」を
    割り当てることで、並列読み込みでも再現性を保つことが可能。
 
    Parameters
    ----------
    seed : int, default=42
        ベースとなるシード値。
 
    Returns
    -------
    Callable[[int], None]
        DataLoader(worker_init_fn=...) に渡すための関数。
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return worker_init_fn
 
 
if __name__ == "__main__":
    # 単体実行時の動作確認用（python seed_utils.py で実行可能）
    set_global_seed(42)
    print("random:", random.random())
    print("numpy:", np.random.rand())
    print("torch:", torch.rand(1).item())