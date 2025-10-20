import warnings

import lightgbm as lgb
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from utils.preprocessing import df_merge, clean_col_names, df_fe
from log_setter.set_up import set_logging

# 警告無視
warnings.filterwarnings("ignore")

# ロギング用意
logger = set_logging("../output/modeling.log")

# データロード及び特徴量エンジニアリング
logger.info("Start merge and feature engineering process")
df = df_merge()
df = df_fe(df)

# デフォルト削除対象カラムを追加
default_del_columns = ["y_class", "net_demand", "theta",
                       "delta", "rental", "return", "station_id"]

# カテゴリカル特徴量を指定
categorical_features = ["parking_hoop", "is_charging_station",
                        "is_renting", "is_installed",
                        "is_returning", "density_category_custom",
                        "facility_type", "day_of_week"]

# カラム整理
X = df.drop(default_del_columns, axis=1)
X = clean_col_names(X)
logger.info(f"Features used for modeling: {X.columns.tolist()}")

# 目的変数
y = df["y_class"]
logger.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# クラスの重み付けを計算
class_weights = y_train.value_counts(normalize=True)
class_weights = 1 / class_weights  # 出力頻度の逆数で重みを作成
class_weights = class_weights / class_weights.mean()  # 正規化して平均1に
logger.info(f"Class weights: {class_weights.to_dict()}")

sample_weight = y_train.map(class_weights)


# LightGBM分類器の初期化
lgb_model = lgb.LGBMRegressor(
    random_state=42,
    objective="regression",
    learning_rate=0.05,
    n_estimators=100,
)
logger.info(f"Initialized LightGBM model: {lgb_model}")

# モデルの訓練
categorical_features_cleaned = [
    col for col in categorical_features if col in X.columns
]
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    categorical_feature=categorical_features_cleaned,
    sample_weight=sample_weight,
    eval_metric="rmse"
)
logger.info("Model training completed")

# 連続値予測
y_pred_reg = lgb_model.predict(X_test)
logger.info("Continuous value prediction complete")

thresholds = [0.5, 1.5]
# クラス予測
y_pred_class = np.digitize(y_pred_reg, bins=thresholds)
logger.info("Class prediction complete")

# 評価指標の準備
qwk = cohen_kappa_score(y_test, y_pred_class, weights="quadratic")
mae = mean_absolute_error(y_test, y_pred_reg)
print(f"QWK: {qwk:.4f}, MAE: {mae:.4f}")
logger.info(f"QWK: {qwk:.4f}, MAE: {mae:.4f}")