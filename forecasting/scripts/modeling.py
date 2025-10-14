import os
import re
import datetime
import holidays
import warnings
import logging

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize
# データ前処理関数をインポート(静的, 動的, 気象のデータを総合的に結合し、前処理を施す関数を用意)
from utils.preprocessing import sample_def, clean_col_names

from log_setter.set_up import set_logging

# 警告無視
warnings.filterwarnings("ignore")

# ロギング用意
logger = set_logging("../output/modeling.log")

# 閾値最適化関数
def optimize_thresholds(y_true, y_pred):
    def func(thresholds):
        thresholds = sorted(thresholds)
        y_pred_class = np.digitize(y_pred, bins=thresholds)
        return -cohen_kappa_score(y_true, y_pred_class, weights="quadratic")
    
    initial_thresholds = [0.5, 1.5]
    result = minimize(func, initial_thresholds, method="Nelder-Mead")
    return sorted(result.x)


# データロード
static_df = pd.read_csv("../../common_data/static.csv")
dynamic_df = pd.read_csv("../../common_data/dynamic.csv")
weather_df = pd.read_csv("../../common_data/weather.csv")

df = sample_df(static_df, dynamic_df, weather_df)

# デフォルト削除対象カラムを追加
default_del_columns = ["y_class", "net_demand", "theta", 
                        "delta", "rental", "return", "station_id"]

# カテゴリカル特徴量を指定
categorical_features = ["parking_hoop", "is_charging_station", "is_renting", 
                        "is_installed", "is_returning", "density_category_custom",
                        "facility_type", "day_of_week"]

# カラム削除
X = df.drop(default_del_columns, axis=1)

X = clean_col_names(X)

# 目的変数
y = df["y_class"]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# クラスの重み付けを計算
class_weights = y_train.value_counts(normalize=True)
class_weights = 1 / class_weights # 出力頻度の逆数で重みを作成
class_weights = class_weights / class_weights.mean() # 正規化して平均1に
print(class_weights)

sample_weight = y_train.map(class_weights)


# LightGBM分類器の初期化
lgb_model = lgb.LGBMRegressor(
    random_state=42,
    objective="regression",
    learning_rate=0.05,
    n_estimators=100, 
)

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


# 連続値予測
y_pred_reg = lgb_model.predict(X_test)

thresholds = optimize_thresholds(y_test, y_pred_reg)
# クラス予測
y_pred_class = np.digitize(y_pred_reg, bins=thresholds)

# 評価指標の準備
qwk = cohen_kappa_score(y_test, y_pred_class, weights="quadratic")
acc = accuracy_score(y_test, y_pred_class)