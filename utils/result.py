import lightgbm as lgb
import matplotlib.pyplot as plt


def plot_data(
        cfg: dict, training_data: dict, output_dir: str,
        model_type: str
) -> None:
    """
    学習過程のロスの遷移を表示する関数

    Parameters
    ----------
    cfg: dict
        実験に使う値のconfigデータ
    training_data: dict
        学習過程のロスのデータ
    output_dir: str
        結果出力先ディレクトリ
    model_type: str
        扱うモデルのタイプ

    Returns
    ----------
    None
    """
    # 学習で得られたデータをまとめる
    if model_type == "LightGBM":
        metric = cfg["params"]["metric"]
    else:
        metric = cfg["Criterion"]
    train_y = [
        item for item in training_data["Train"][metric]
    ]
    valid_y = [
        item for item in training_data["Valid"][metric]
    ]
    x = [i + 1 for i in range(len(train_y))]

    # 画像のスタイルを指定する
    plt.figure(figsize=(18, 12))
    plt.title("Loss comparison", size=15, color="red")
    plt.grid()

    # データをプロット
    plt.plot(x, train_y, label="Train")
    plt.plot(x, valid_y, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 画像を保存する
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curve.png")


def extract_feature_importance(
    model: lgb.basic.Booster, out_dir: list[str]
) -> None:
    """
    LightGBMモデルの特徴量重要度を抽出する関数

    Parameters
    ----------
    model: lgb.basic.Booster
        実験で学習させたモデルのインスタンス
    out_dir: str
        結果出力先ディレクトリ

    Retturns
    ----------
    None
    """
    # 画像のスタイルを指定する
    plt.figure(figsize=(18, 12))
    plt.title("LightGBM Feature Importance", size=15, color="red")
    plt.grid()

    # 特徴量の重要度を描画
    lgb.plot_importance(model)

    # 画像を保存
    plt.savefig(f"{out_dir}/feature_importance.png")