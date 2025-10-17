import re
import json

import numpy as np
import pandas as pd


# カテゴリ辞書をクラス分けする関数
def classify_facility(name, facility_dict):
    for category, keywords in facility_dict.items():
        for kw in keywords:
            if kw in str(name):
                return category
    return "other"


def fill_weather_code(row):
    # 既に天気コードが入っていればそのまま返す
    if pd.notnull(row["Weather"]):
        return row["Weather"]

    rain = row.get("Precipitation(mm)", np.nan)
    temp = row.get("Temperature(℃)", np.nan)

    if pd.notnull(rain) and rain > 0:
        if pd.notnull(temp) and temp <= 2:
            return 14  # 雪
        else:
            return 12  # 雨

    if pd.notnull(rain) and rain == 0:
        return 3  # 曇

    # 上記以外（降水量が欠損など）
    return 99


def json_data_load(json_path):
    """jsonファイルを読み込む関数"""

    with open(json_path, "r", encoding="utf-8") as f:
        df = json.load(f)

    df = df["data"]["stations"]

    return df


# 閾値の設定関数
def categorize_demand(row):
    # 多すぎ
    if row["net_demand"] <= -row["theta"]:
        return 2
    # 少なすぎ
    elif row["net_demand"] >= row["theta"]:
        return 0
    # 適切
    else:
        return 1
    

def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r"[^A-Za-z0-9_]+", "", col)
        new_cols.append(new_col)
    df.columns = new_cols
    return df