"""
train_irl.py
=====================================================================
【概要】
collect_demo.pyで集めたデモデータ(demo_data.pt)を使い、
TrajectoryTransformerの policy_head（行動選択）と reward_head（IRL報酬関数）
を同時に学習させるスクリプト。
 
【損失関数の設計（重要）】
 
(A) 行動クローニング損失 (BC Loss / Behavioral Cloning Loss)
    policy_headが出力する行動ロジット(5クラス分)に対し、
    エキスパートが実際に選んだ行動を正解ラベルとした
    クロスエントロピー損失を計算する。
    「同じ状況ならエキスパートと同じ行動を選ぶ確率を最大化する」という、
    最もシンプルな模倣学習(Behavioral Cloning)の考え方そのもの。
 
(B) IRL報酬損失 (IRL Reward Loss)
    reward_headに対しては、"ある状態の下で5つの行動それぞれを取った場合の
    報酬値"をすべて計算し、それを5クラス分の"ロジット"とみなして
    クロスエントロピー損失を取る。
 
    これは Maximum Entropy IRL（最大エントロピー逆強化学習）の
    考え方を簡略化したものである。MaxEnt IRLでは、
        方策 π(a|s) ∝ exp(reward(s, a))
    つまり「報酬が高い行動ほど選ばれやすい」というソフトマックス方策を
    仮定し、エキスパートの行動が持つ尤度を最大化するように報酬関数を学習する。
    これはまさに「reward(s, a)をロジットとみなしたソフトマックス分類問題」
    として定式化でき、CrossEntropyLossでそのまま実装できる。
 
    数式的に書くと：
        L_irl = -log( exp(reward(s, a_expert)) / Σ_a' exp(reward(s, a')) )
    これは reward_all_actions を「ロジット」として
    nn.CrossEntropyLoss(reward_all_actions, a_expert) を計算するのと等価。
 
(C) 総合損失
        total_loss = bc_loss + IRL_LOSS_WEIGHT * irl_loss
    2つの損失は「異なるヘッド（policy_head / reward_head）」に対する
    勾配を生み出すが、共通のTransformerエンコーダ部分（状態表現）を
    通して間接的に影響し合う。これにより、Transformerエンコーダ自身も
    「行動選択に有用」かつ「報酬推定に有用」な状態表現を学習するように
    最適化される。
=====================================================================
"""

# 必要なライブラリをインポート
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, WeightedRandomSampler
# 自作のファイルをインポート
import config
from seed_utils import set_global_seed, get_seeded_worker_init_fn
from transformer_planner import TrajectoryTransformer, NUM_ACTIONS, ACTION_NAMES


# =====================================================================
# クラス不均衡対策：Class Weights（クラス重み付け）の計算
# =====================================================================
def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    行動ラベルの出現頻度から、CrossEntropyLossに渡すクラス重みを計算する。

    【考え方（"balanced"方式）】
        weight_c = 全サンプル数 / (クラス数 × クラスcの出現数)

    出現数が少ないクラス（＝collect_demo.pyのオーバーサンプリング後でも
    まだ少数派になりがちなLANE_LEFT/LANE_RIGHT）ほど重みが大きくなり、
    そのクラスを間違えたときの損失（＝勾配）が相対的に強調される。
    逆にIDLEのような多数派クラスは重みが1未満になり、
    「多数派を当てておけば損失が下がる」という学習の近道に
    モデルが逃げ込むのを防ぐ。

    Parameters
    ----------
    labels : torch.Tensor
        shape = [N]、dtype=long の行動ラベル列
        （収集データ全体ではなく、必ず"学習に使う分割"のラベルから
          計算すること。検証データの分布を重みに混ぜると、
          間接的に検証データの情報が学習に漏れてしまうため）
    num_classes : int
        行動の総数（=NUM_ACTIONS）

    Returns
    -------
    torch.Tensor
        shape = [num_classes] のクラス重みベクトル。
        nn.CrossEntropyLoss(weight=...) にそのまま渡せる。
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    # 出現数0のクラスがあった場合の0除算を防ぐため、下限1でクリップする
    counts_clamped = torch.clamp(counts, min=1.0)
    weights = counts_clamped.sum() / (num_classes * counts_clamped)
    return weights


# =====================================================================
# クラス不均衡対策：クラスごとの評価指標（Precision / Recall / F1）
# =====================================================================
def compute_confusion_matrix(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    予測ラベルと正解ラベルから混同行列を計算する。

    行(dim=0)=正解クラス、列(dim=1)=予測クラス。
    cm[i, j] = 「正解がiなのに、jと予測した」件数。
    対角成分 cm[i, i] が正しく予測できた件数。

    Parameters
    ----------
    preds : torch.Tensor
        shape = [N]、モデルの予測ラベル
    labels : torch.Tensor
        shape = [N]、正解ラベル
    num_classes : int

    Returns
    -------
    torch.Tensor
        shape = [num_classes, num_classes] の混同行列（long型）
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    indices = labels.long() * num_classes + preds.long()
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    cm += counts.view(num_classes, num_classes)
    return cm


def compute_per_class_metrics(cm: torch.Tensor) -> dict:
    """
    混同行列から、クラスごとのPrecision/Recall/F1と、
    全クラス単純平均（マクロ平均）を計算する。

    【なぜ"全体の正解率(accuracy)"だけでは不十分か（重要）】
    accuracyは「全サンプルのうち何件正しく当てたか」なので、
    IDLEのような多数派クラスさえ当てておけば高い値が出てしまう。
    極端な話、「常にIDLEとだけ予測するモデル」でも
    accuracyは(IDLE比率)%前後になり得るが、そんなモデルは
    LANE_LEFT/RIGHTを一度も正しく選べていない、実用上は失格のモデルである。

    そこで、クラスごとのRecall（そのクラスの正解サンプルのうち、
    実際に正しくそのクラスだと予測できた割合）をマクロ平均（クラス数で
    単純平均、件数で重み付けしない）することで、
    「少数派クラスをちゃんと拾えているか」を平等に評価できるようにする。

    Parameters
    ----------
    cm : torch.Tensor
        compute_confusion_matrix()で得た混同行列

    Returns
    -------
    dict
        {class_id: {"precision":.., "recall":.., "f1":.., "support":..}, ...,
         "macro_recall": .., "macro_precision": .., "macro_f1": ..}
    """
    num_classes = cm.size(0)
    per_class = {}
    recalls, precisions, f1s = [], [], []

    for c in range(num_classes):
        tp = cm[c, c].item()
        support = cm[c, :].sum().item()          # このクラスの正解サンプル総数
        predicted_as_c = cm[:, c].sum().item()    # このクラスと予測された総数
        fn = support - tp
        fp = predicted_as_c - tp

        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted_as_c if predicted_as_c > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        per_class[c] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    per_class["macro_recall"] = sum(recalls) / num_classes
    per_class["macro_precision"] = sum(precisions) / num_classes
    per_class["macro_f1"] = sum(f1s) / num_classes
    return per_class


def format_per_class_metrics(metrics: dict, action_names: list) -> str:
    """per_class_metricsの結果を表形式の文字列に整形する。"""
    lines = [f"    {'行動':>10s} | {'Precision':>9s} | {'Recall':>7s} | "
             f"{'F1':>6s} | {'support':>7s}"]
    for c, name in enumerate(action_names):
        m = metrics[c]
        lines.append(
            f"    {name:>10s} | {m['precision']*100:8.1f}% | "
            f"{m['recall']*100:6.1f}% | {m['f1']*100:5.1f}% | {m['support']:7d}"
        )
    lines.append(
        f"    {'macro avg':>10s} | {metrics['macro_precision']*100:8.1f}% | "
        f"{metrics['macro_recall']*100:6.1f}% | {metrics['macro_f1']*100:5.1f}% |"
    )
    return "\n".join(lines)


def compute_rewards_for_all_actions(
    model: TrajectoryTransformer, state_repr: torch.Tensor
) -> torch.Tensor:
    """
    ある状態表現に対して、5つの行動それぞれについて reward_head の出力を
    計算し、[batch_size, num_actions] の"報酬ロジット"として返す関数。
 
    Parameters
    ----------
    model : TrajectoryTransformer
        reward_headを含む学習対象モデル
    state_repr : torch.Tensor
        shape = [batch_size, d_model]
        forward()で得られた状態表現ベクトル
 
    Returns
    -------
    torch.Tensor
        shape = [batch_size, num_actions]
        各行動を取った場合の報酬値を並べたテンソル
        （そのままCrossEntropyLossのlogits引数として使える）
    """
    batch_size = state_repr.size(0)
    device = state_repr.device
 
    reward_list = []
    for action_id in range(NUM_ACTIONS):
        # このバッチ全体に対して「行動action_idを取った」場合のone-hot
        # ベクトルを作る。shape: [batch_size, num_actions]
        # （バッチ内の全サンプルに同じ行動one-hotを適用している点に注意。
        #   「バッチの全サンプルについて"もし行動action_idを取ったら
        #   報酬はいくつになるか"」を一括計算している）
        onehot = torch.zeros(batch_size, NUM_ACTIONS, device=device)
        onehot[:, action_id] = 1.0
 
        # reward_head(state, action=onehot) -> shape: [batch_size, 1]
        reward = model.compute_reward(state_repr, onehot)
        reward_list.append(reward)
 
    # 5つの [batch_size, 1] テンソルを行動次元(dim=1)で連結する
    # → shape: [batch_size, num_actions]
    rewards_all_actions = torch.cat(reward_list, dim=1)
    return rewards_all_actions
 
 
def run_one_epoch(
    model: TrajectoryTransformer,
    data_loader: DataLoader,
    bc_criterion: nn.Module,
    irl_criterion: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    collect_predictions: bool = False,
) -> dict:
    """
    1エポック分の学習（optimizerが指定された場合）または評価
    （optimizer=Noneの場合）を実行する共通関数。
 
    学習と検証でほぼ同じ処理（順伝播 → 損失計算）を繰り返すため、
    コードの重複を避けるために関数化している。
 
    Parameters
    ----------
    model : TrajectoryTransformer
    data_loader : DataLoader
        (obs_batch, action_batch) のペアを返すDataLoader
    bc_criterion : nn.Module
        行動クローニング損失（通常はnn.CrossEntropyLoss()）
    irl_criterion : nn.Module
        IRL報酬損失（通常はnn.CrossEntropyLoss()）
    optimizer : torch.optim.Optimizer, optional
        Noneでなければ学習モード（逆伝播とパラメータ更新を行う）
        Noneであれば評価モード（torch.no_grad()で勾配計算をしない）
    collect_predictions : bool, default=False
        Trueの場合、全バッチの予測ラベル・正解ラベルを1本のテンソルに
        まとめて返す（クラスごとのPrecision/Recall計算に使う）。
        学習フェーズでは不要な追加コストのため、通常はFalseのまま使う。
 
    Returns
    -------
    dict
        {"bc_loss": float, "irl_loss": float, "accuracy": float,
         "predictions": torch.Tensor (collect_predictions=Trueの場合のみ),
         "labels": torch.Tensor (collect_predictions=Trueの場合のみ)}
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
 
    bc_loss_sum = 0.0
    irl_loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = [] if collect_predictions else None
    all_labels = [] if collect_predictions else None
 
    # 学習時のみ勾配計算を有効にする（torch.set_grad_enabled で切り替え）
    with torch.set_grad_enabled(is_training):
        for obs_batch, action_batch in data_loader:
            # obs_batch: [B, SEQ_LEN, feature_dim]
            # action_batch: [B]（エキスパートの行動ラベル、long型）
 
            if is_training:
                optimizer.zero_grad()
 
            # --- (1) モデルにバッチを入力し、行動ロジットと状態表現を得る ---
            # action_logits: [B, num_actions], state_repr: [B, d_model]
            action_logits, state_repr = model(obs_batch)
 
            # --- (2) BC損失：エキスパート行動を正解ラベルとした分類損失 ---
            bc_loss = bc_criterion(action_logits, action_batch)
 
            # --- (3) IRL損失：reward_headに5行動分の報酬を計算させ、
            #         エキスパート行動の報酬が最も高くなるよう学習する ---
            rewards_all_actions = compute_rewards_for_all_actions(model, state_repr)
            # rewards_all_actions: [B, num_actions]
            irl_loss = irl_criterion(rewards_all_actions, action_batch)
 
            # --- (4) 2つの損失を重み付けして合算する ---
            total_loss = bc_loss + config.IRL_LOSS_WEIGHT * irl_loss
 
            if is_training:
                # 逆伝播でモデル全体（Transformer本体 + 両方のhead）の
                # 勾配を計算し、パラメータを更新する
                total_loss.backward()
                optimizer.step()
 
            batch_size = obs_batch.size(0)
            bc_loss_sum += bc_loss.item() * batch_size
            irl_loss_sum += irl_loss.item() * batch_size
 
            predicted = torch.argmax(action_logits, dim=-1)
            correct += (predicted == action_batch).sum().item()
            total += batch_size

            if collect_predictions:
                all_preds.append(predicted.detach())
                all_labels.append(action_batch.detach())
 
    result = {
        "bc_loss": bc_loss_sum / max(1, total),
        "irl_loss": irl_loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
    }
    if collect_predictions:
        result["predictions"] = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, dtype=torch.long)
        result["labels"] = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    return result
 
 
def train() -> None:
    """
    デモデータを読み込み、TrajectoryTransformerをBC損失+IRL損失で
    学習させ、検証データで最も性能の良かった重みを保存するメイン関数。
    """
    set_global_seed(config.SEED)
 
    # ------------------------------------------------------------
    # ① デモデータを読み込む
    # ------------------------------------------------------------
    demo = torch.load(config.DEMO_DATA_PATH)
    observations = demo["observations"]   # [N, SEQ_LEN, feature_dim]
    actions = demo["actions"]             # [N]
    feature_dim = demo["feature_dim"]
    seq_len = demo["seq_len"]
 
    print(f"[train_irl] デモデータ読み込み完了: "
          f"observations.shape={tuple(observations.shape)}, "
          f"actions.shape={tuple(actions.shape)}")
 
    # collect_demo.pyとconfig.pyの設定がズレていないかを確認する
    # （SEQ_LENが違うと、学習済みモデルを推論時に正しく使えなくなるため）
    assert seq_len == config.SEQ_LEN, (
        f"デモデータのSEQ_LEN({seq_len})とconfig.SEQ_LEN({config.SEQ_LEN})が"
        f"一致していません。config.py変更後は collect_demo.py を"
        f"再実行してください。"
    )
 
    # ------------------------------------------------------------
    # ② train/valに分割する
    # ------------------------------------------------------------
    dataset = TensorDataset(observations, actions)
    val_size = max(1, int(len(dataset) * config.VAL_SPLIT_RATIO))
    train_size = len(dataset) - val_size
 
    # 再現性のため、分割にもシード固定済みのgeneratorを使う
    split_generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )
 
    # DataLoaderのシャッフルにも専用generatorを渡すことで、
    # 「何エポック目にどの順でバッチが渡されるか」まで再現可能にしている
    loader_generator = torch.Generator().manual_seed(config.SEED)

    # ------------------------------------------------------------
    # ②' クラス不均衡対策：学習分割(train_dataset)のラベル分布を先に取得する
    #     （Class Weights・WeightedRandomSamplerの両方で使うため、
    #       DataLoader構築より前に計算しておく）
    # ------------------------------------------------------------
    train_labels = actions[train_dataset.indices]
    balanced_weights = compute_class_weights(train_labels, NUM_ACTIONS)

    if config.USE_WEIGHTED_SAMPLER:
        # 【WeightedRandomSamplerとClass Weighted Lossの役割の違い】
        # ・Class Weighted Loss（後述）は「間違えたときの罰則の大きさ」を
        #   クラスごとに変える。ミニバッチの中身自体は変わらない。
        # ・WeightedRandomSamplerは「そもそもミニバッチに少数派クラスが
        #   より高い確率で含まれるようにする」仕組み。1バッチの中に
        #   LANE_LEFT/RIGHTが全く含まれない、といった事態を減らせるため、
        #   損失の重み付けだけでは是正しきれなかった"学習シグナルの薄さ"に
        #   直接効く。
        # 2つを併用すると効果が過剰に重なる可能性があるため、
        # サンプラー側は素の(balanced)重みを使い、損失側は
        # config.CLASS_WEIGHT_LOSS_SCALEで強さを調整できるようにしている。
        sample_weights = balanced_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,  # samplerを使う場合、shuffle=Trueとは併用不可
            worker_init_fn=get_seeded_worker_init_fn(config.SEED),
        )
        print("[train_irl] WeightedRandomSamplerを有効化: "
              "各ミニバッチにLANE_LEFT/RIGHTがより高い確率で含まれるようにします。")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            generator=loader_generator,
            worker_init_fn=get_seeded_worker_init_fn(config.SEED),
        )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
 
    print(f"[train_irl] train={train_size}件, val={val_size}件")

    # ------------------------------------------------------------
    # ②'' クラス不均衡対策：CrossEntropyLossに渡すClass Weightsを決める
    #     （collect_demo.py側で比率を底上げしていても、完全に均等には
    #       ならないことが多いため、学習側でも二重に手当てする）
    #     CLASS_WEIGHT_LOSS_SCALEで重みの強さ（1.0からの乖離）を
    #     調整できるようにしている。WeightedRandomSamplerと併用する場合、
    #     効果が重なりすぎないよう1.0未満（例:0.5）に弱めることを推奨する。
    # ------------------------------------------------------------
    if config.USE_CLASS_WEIGHTS:
        scale = config.CLASS_WEIGHT_LOSS_SCALE
        class_weights = 1.0 + (balanced_weights - 1.0) * scale
        print(f"[train_irl] Class Weights（訓練データの分布から算出、"
              f"scale={scale}）:")
        for name, w, c in zip(
            ACTION_NAMES, class_weights.tolist(),
            torch.bincount(train_labels, minlength=NUM_ACTIONS).tolist(),
        ):
            print(f"    {name:>10s}: weight={w:.3f}  (train内 {c}件)")
    else:
        class_weights = None
        print("[train_irl] config.USE_CLASS_WEIGHTS=False のため、"
              "均等重みでクロスエントロピー損失を計算します。")

    # ------------------------------------------------------------
    # ③ モデル・損失関数・最適化手法を初期化する
    # ------------------------------------------------------------
    model = TrajectoryTransformer(
        feature_dim=feature_dim,
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        num_actions=NUM_ACTIONS,
        max_seq_len=config.SEQ_LEN,
    )
 
    # (A) 行動クローニング損失：policy_headの出力に対する多クラス分類の
    #     クロスエントロピー損失
    #     class_weightsを渡すことで、LANE_LEFT/RIGHTのような少数派の
    #     行動を間違えたときの損失（＝勾配）をより強く反映させる
    bc_criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    # (B) IRL報酬損失：reward_headの出力(5行動分)に対する
    #     クロスエントロピー損失
    #     こちらも同じクラス不均衡の影響を受けるため、同じ重みを適用する
    irl_criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
 
    # ------------------------------------------------------------
    # ④ 学習ループ
    # ------------------------------------------------------------
    print(f"\n[train_irl] 学習開始 "
          f"(epochs={config.NUM_EPOCHS}, batch_size={config.BATCH_SIZE}, "
          f"lr={config.LEARNING_RATE}, IRL_LOSS_WEIGHT={config.IRL_LOSS_WEIGHT})")
    print("-" * 100)
 
    best_val_macro_f1 = -1.0
    best_val_total_loss_at_best = None
    best_epoch = -1
    best_metrics = None
 
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # --- 学習フェーズ ---
        train_stats = run_one_epoch(
            model, train_loader, bc_criterion, irl_criterion, optimizer=optimizer
        )
 
        # --- 検証フェーズ（optimizer=Noneを渡すことで評価モードになる） ---
        # collect_predictions=Trueにして、クラスごとのRecall/Precisionを
        # 計算できるようにする
        val_stats = run_one_epoch(
            model, val_loader, bc_criterion, irl_criterion, optimizer=None,
            collect_predictions=True,
        )
 
        val_total_loss = val_stats["bc_loss"] + config.IRL_LOSS_WEIGHT * val_stats["irl_loss"]

        cm = compute_confusion_matrix(
            val_stats["predictions"], val_stats["labels"], NUM_ACTIONS
        )
        metrics = compute_per_class_metrics(cm)
        lane_left_recall = metrics[ACTION_NAMES.index("LANE_LEFT")]["recall"]
        lane_right_recall = metrics[ACTION_NAMES.index("LANE_RIGHT")]["recall"]
 
        print(
            f"[epoch {epoch:3d}/{config.NUM_EPOCHS}] "
            f"train: BC={train_stats['bc_loss']:.4f} "
            f"IRL={train_stats['irl_loss']:.4f} "
            f"acc={train_stats['accuracy']*100:5.1f}% | "
            f"val: BC={val_stats['bc_loss']:.4f} "
            f"IRL={val_stats['irl_loss']:.4f} "
            f"acc={val_stats['accuracy']*100:5.1f}% "
            f"macro_recall={metrics['macro_recall']*100:5.1f}% "
            f"(L_LEFT R={lane_left_recall*100:5.1f}%, L_RIGHT R={lane_right_recall*100:5.1f}%)"
        )
 
        # --- モデル選定基準：val_total_loss ではなく macro_recall を使う ---
        # 【重要】以前の実装は「val_total_lossが最小のエポック」を保存して
        # いたが、これは"IDLEなど多数派クラスさえ当てておけば下がる"指標で
        # あり、LANE_LEFT/RIGHTの再現率を全く見ていなかった。
        # 結果として「全体の損失は低いが、車線変更はほとんど提案しない」
        # 保守的なエポックが選ばれてしまう恐れがある。
        # そこで、全クラスのRecallを均等に平均したmacro_recallが
        # 最大のエポックを採用することで、「少数派クラスも含めて
        # バランス良く判断できるモデル」を優先的に選ぶようにする。
        if metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = metrics["macro_f1"]
            best_val_total_loss_at_best = val_total_loss
            best_epoch = epoch
            best_metrics = metrics
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
 
    print("-" * 100)
    print(f"[train_irl] 学習完了。最良モデル "
          f"(epoch={best_epoch}, macro_f1={best_val_macro_f1*100:.1f}%, "
          f"macro_recall={best_metrics['macro_recall']*100:.1f}%, "
          f"val_total_loss={best_val_total_loss_at_best:.4f}) を "
          f"{config.MODEL_SAVE_PATH} に保存しました。")
    print("\n[train_irl] 採用したエポックにおける、検証データでの"
          "クラスごとの性能内訳:")
    print(format_per_class_metrics(best_metrics, ACTION_NAMES))
    print(f"\n[train_irl] main.py を実行すると、この学習済みモデルが"
          f"自動的に読み込まれます。")
    print(f"[train_irl] 【診断のヒント】上表のLANE_LEFT/LANE_RIGHTのRecallが"
          f"低い場合、検証データの時点で少数派クラスを正しく判定できていない"
          f"ということなので、①collect_demo.pyでの本物の車線変更サンプル数を"
          f"さらに増やす、②config.CLASS_WEIGHT_LOSS_SCALEを上げる、"
          f"③config.NUM_EPOCHSを増やす、のいずれかを試してください。\n"
          f"逆にRecallは高いのに main.py 実行時の行動分布（前回の対策で"
          f"追加したログ）で車線変更がほとんど選ばれない場合は、学習データの"
          f"分布と実際の走行時の分布がズレている可能性が高いため、"
          f"main.py / visualize.py の env.configure() が collect_demo.py と"
          f"完全に一致しているか（lanes_count, vehicles_densityなど）を"
          f"必ず確認してください。")
 
 
if __name__ == "__main__":
    train()