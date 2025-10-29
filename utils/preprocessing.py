import os
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from .other_function import json_data_load


# カテゴリ辞書をクラス分けする関数
def classify_facility(name):
    # カテゴリ辞書
    facility_dict = {
        "station": ["駅", "出口", "東口", "西口", "北口", "南口", "改札"],
        "convenience": ["セブンイレブン", "ファミリーマート", "ローソン", "ミニストップ"],
        "supermarket": ["イオン", "マルエツ", "スーパー", "西友", "ライフ", "店"],
        "residential": ["マンション", "団地", "レジデンス", "ハイツ", "アパート"],
        "public": ["区役所", "図書館", "体育館", "ホール", "市場", "保健"],
        "school": ["大学", "高校", "中学校", "小学校", "学園", "幼稚園"],
        "park_tourism": ["公園", "東京タワー", "スカイツリー", "観光", "博物館", "美術館", "ホテル"],
        "office": ["ビル", "オフィス", "会社", "社"]
    }

    for category, keywords in facility_dict.items():
        for kw in keywords:
            if kw in str(name):
                return category
    return "other"


def fill_weather_code(row):
    # "Weather"がSeriesの場合は最初の値を取得
    weather = row.get("Weather", np.nan)
    if isinstance(weather, pd.Series):
        weather = weather.values[0]
    if pd.notnull(weather):
        return weather

    rain = row.get("Precipitation(mm)", np.nan)
    if isinstance(rain, pd.Series):
        rain = rain.values[0]
    temp = row.get("Temperature(℃)", np.nan)
    if isinstance(temp, pd.Series):
        temp = temp.values[0]

    if pd.notnull(rain) and rain > 0:
        if pd.notnull(temp) and temp <= 2:
            return 14  # 雪
        return 12  # 雨

    if pd.notnull(rain) and rain == 0:
        return 3  # 曇

    return 99  # 上記以外（降水量が欠損など）


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


def static_preprocessing():
    """静的データ前処理関数"""
    json_path = "../../common_data/static.json"
    df = json_data_load(json_path)
    df = pd.DataFrame(df)

    df = df[df["address"].str.contains("東京都", na=False)]
    df = df[df["address"].str.contains("区", na=False)]
    # addressごとにの区名を抽出し、駅数をカウント
    df["ward"] = df["address"].str.extract(r"東京都(.+?)区", expand=False)
    df["ward_count"] = df["ward"].map(df["ward"].value_counts())
    # 抽出した区の名前の分布をカウント
    ward_counts = df['ward'].value_counts()

    df["vehicle_capacity_numeric"] = pd.to_numeric(
        df["vehicle_capacity"], errors="coerce"
    )

    data = {
        "ward": [
            "千代田", "中央", "港", "新宿", "文京", "台東", "墨田", "江東",
            "品川", "目黒", "大田", "世田谷", "渋谷", "中野", "杉並", "豊島",
            "北", "荒川", "板橋", "練馬", "足立", "葛飾", "江戸川"
        ],
        "area_km2": [
            11.66, 10.09, 20.37, 18.22, 11.29, 10.11, 27.77, 40.16,
            22.84, 14.67, 60.66, 58.25, 15.11, 15.59, 34.02, 13.01,
            20.59, 10.20, 32.22, 48.07, 53.25, 34.80, 49.90
        ]
    }

    ward_area_df = pd.DataFrame(data)

    ward_counts_df = ward_counts.reset_index()
    # ward_counts_dfの"index"を"ward", "ward_count"を"station_count"にリネーム
    ward_counts_df.columns = ['ward', 'station_count']

    # 型を揃える
    ward_counts_df["ward"] = ward_counts_df["ward"].astype(str)
    ward_area_df["ward"] = ward_area_df["ward"].astype(str)

    # ward_counts_dfとward_area_dfを結合
    ward_counts_with_area = pd.merge(
        ward_counts_df, ward_area_df, on="ward", how="left"
    )

    ward_counts_with_area["station_density"] = (
        ward_counts_with_area["station_count"]
        / ward_counts_with_area["area_km2"]
    )

    max_density = ward_counts_with_area["station_density"].max()
    bins = [0, 2.0, 4.0]
    if max_density > 4.0:
        bins.append(max_density)
    else:
        bins.append(4.0)

    # 重複を除去し昇順に
    bins = sorted(set(bins))

    # labels数をbins数-1に合わせる
    labels = [str(i) for i in range(len(bins) - 1)]

    ward_counts_with_area["density_category_custom"] = pd.cut(
        ward_counts_with_area["station_density"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
        duplicates="drop"
    )

    df = pd.merge(df, ward_counts_with_area, on="ward", how="left")

    df["facility_type_label"] = df["name"].apply(classify_facility)

    le = LabelEncoder()
    df["facility_type"] = le.fit_transform(df["facility_type_label"])

    drop_columns = [
        "name", "address", "rental_uris",
        "parking_type", "contact_phone", "vehicle_capacity",
        "ward", "area_km2", "facility_type_label"
    ]
    df = df.drop(drop_columns, axis=1)

    return df


def dynamic_preprocessing():
    """動的データ前処理関数"""
    df = pd.read_csv("../../common_data/odpt_20251024.csv")

    df["last_reported_datetime"] = pd.to_datetime(
        df["last_reported"], unit="s")

    columns_to_drop = [
        "vehicle_docks_available",
        "vehicle_types_available",
        "last_reported"
    ]
    df = df.drop(columns=columns_to_drop)

    df["month"] = df["last_reported_datetime"].dt.month
    df["date"] = df["last_reported_datetime"].dt.day
    df["hours"] = df["last_reported_datetime"].dt.hour

    # month, date, hoursを数値型に
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")

    df = df.drop(columns=["last_reported_datetime"])

    return df


def weather_preprocessing():
    """気象データ前処理関数"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    weather_path = os.path.join(base_dir, "common_data", "weather_data_.csv")
    df = pd.read_csv(
        weather_path,
        header=None,
        encoding='shift-jis',
        sep='|',
        engine='python'
    )

    # 2. 単一の列をコンマで分割し、新しいDataFrameを作成
    df = df[0].str.split(',', expand=True)

    # 3. 実際のデータフレームを作成し、インデックス5からデータを取得
    df_data = df.iloc[5:].copy()

    # 4. 保持すべき列のインデックスを特定 (0-indexed)
    columns_to_keep_indices = [0, 1, 2, 3, 4, 8, 11, 13, 16]

    # 5. KEEPする列のみを選択
    df = df_data.iloc[:, columns_to_keep_indices]

    # 6. カラム名を英語にリネーム (9要素)
    clean_names = [
        'year', 'month', 'date', 'hours', 'precipitation_mm',
        'temperature_c', 'wind_speed_ms', 'wind_direction', 'weather_code'
    ]
    df.columns = clean_names

    # 7. データ型の変換
    cols_to_convert = [
        'year', 'month', 'date',
        'hours', 'precipitation_mm', 'temperature_c', 'wind_speed_ms'
    ]
    for col in cols_to_convert:
        if col in df.columns:
            # エラーを無視してNaNに変換し、数値型に変換
            df[col] = pd.to_numeric(df[col], errors='coerce')

    wind_direction_map = {
        "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5,
        "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5,
        "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5,
        "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5,
        "静穏": 999
    }
    df["wind_direction"] = df["wind_direction"].map(wind_direction_map)
    df["weather_code"] = df.apply(fill_weather_code, axis=1)

    return df


def df_merge():
    """データフレームの結合関数"""
    static_df = static_preprocessing()
    dynamic_df = dynamic_preprocessing()
    weather_df = weather_preprocessing()

    # station_idの型を揃える
    static_df["station_id"] = static_df["station_id"].astype(str)
    dynamic_df["station_id"] = dynamic_df["station_id"].astype(str)

    # 静的データと動的データを結合
    odpt_df = pd.merge(static_df, dynamic_df, on="station_id", how="left")
    del_cols = ["Unnamed: 0"]
    odpt_df.drop(del_cols, axis=1, inplace=True)

    df = pd.merge(
        odpt_df, weather_df,
        on=["month", "date", "hours"],
        how="inner"
    )

    # LightGBM用に型変換
    for col in ["is_renting", "is_installed", "is_returning"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


def df_fe(df):
    """結合データの特徴量エンジニアリング関数"""

    # bike_ratio: 貸出可能自転車の割合
    df["bike_ratio"] = df["num_bikes_available"] / (
        df["num_bikes_available"] + df["num_docks_available"]
        )
    # dock_ratio: 返却可能ドックの割合
    df["dock_ratio"] = df["num_docks_available"] / (
        df["num_bikes_available"] + df["num_docks_available"]
        )
    # 空満判定特徴量
    df["is_empty"] = (
        df["num_bikes_available"] == 0)
    df["is_full"] = (
        df["num_docks_available"] == 0)

    # 曜日に関する特徴量
    df["datetime"] = pd.to_datetime(dict(
        year=2025,
        month=df["month"],
        day=df["date"],
        hour=df["hours"]
    ))
    # 曜日を作成
    df["day_of_week"] = df["datetime"].dt.weekday

    # net_demandの計算
    # datetimeを明示的にdatetime型に
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ソート
    df = df.sort_values(["station_id", "datetime"])

    # 差分計算
    df["delta"] = df.groupby("station_id")["num_bikes_available"].diff()
    df["delta"] = df["delta"].fillna(df["num_bikes_available"])

    # 貸出と返却の分解
    df["rental"] = df["delta"].apply(
        lambda x: -x if pd.notna(x) and x < 0 else 0)
    df["return"] = df["delta"].apply(
        lambda x: x if pd.notna(x) and x > 0 else 0)

    # 純需要
    df["net_demand"] = df["rental"] - df["return"]

    # 閾値の設定
    theta_ratio = 0.1
    df["theta"] = df["vehicle_capacity_numeric"] * theta_ratio
    df["y_class"] = df.apply(categorize_demand, axis=1)

    df = clean_col_names(df)

    return df
