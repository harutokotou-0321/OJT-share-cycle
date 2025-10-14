import warnings

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from other_function import (
    classify_facility,
    fill_weather_code,
    json_data_load,
    categorize_demand,
    clean_col_names
)

# 警告無視
warnings.filterwarnings("ignore")


def static_preprocessing(df):
    """静的データ前処理関数"""
    json_path = "../../common_data/static.json"
    df = json_data_load(json_path)

    df = df[df["address"].str.contains("東京都", na=False)]
    df = df[df["address"].str.contains("区", na=False)]
    df["ward"] = df["address"].str.extract(r"東京都(.+?)区", expand=False)
    df["ward_count"] = df["ward"].value_counts()

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

    ward_counts_df = df["ward_count"].reset_index()
    ward_counts_df.columns = ["ward", "station_count"]
    ward_counts_with_area = pd.merge(
        ward_counts_df, ward_area_df, on="ward", how="left"
    )

    ward_counts_with_area["station_density"] = (
        ward_counts_with_area["station_count"] / ward_counts_with_area["area_km2"]
    )

    bins = [0, 2.0, 4.0, ward_counts_with_area["station_density"].max()]
    labels = ["0", "1", "2"]

    ward_counts_with_area["density_category_custom"] = pd.cut(
        ward_counts_with_area["station_density"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
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


def dynamic_preprocessing(df):
    """動的データ前処理関数"""
    df = json_data_load("../../common_data/dynamic.json")

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
    df = df.drop(columns=["last_reported_datetime"])

    return df


def weather_preprocessing(df):
    """気象データ前処理関数"""
    df.columns = [
        "_".join([str(c) for c in col if str(c) != "nan"]).strip()
        for col in df.columns.values
    ]

    cleaned_columns = []
    for col in df.columns:
        cleaned_columns.append(col.split("_", 1)[0])

    japanese_to_english_map = {
        "年月日時": "DateTime",
        "降水量(mm)": "Precipitation(mm)",
        "気温(℃)": "Temperature(℃)",
        "風速(m/s)": "WindSpeed(m/s)",
        "風向": "WindDirection",
        "天気": "Weather"
    }
    df.columns = [
        japanese_to_english_map.get(col, col) for col in cleaned_columns
    ]

    wind_direction_map = {
        "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5,
        "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5,
        "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5,
        "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5,
        "静穏": np.nan
    }
    df["WindDirection"] = df["WindDirection"].map(wind_direction_map)

    df["Weather"] = df.apply(fill_weather_code, axis=1)

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["month"] = df["DateTime"].dt.month
    df["date"] = df["DateTime"].dt.day
    df["hours"] = df["DateTime"].dt.hour
    df = df.drop(columns=["DateTime"])

    return df


def df_merge():
    """データフレームの結合関数"""
    static_df = static_preprocessing(pd.DataFrame())
    dynamic_df = dynamic_preprocessing(pd.DataFrame())
    weather_df = weather_preprocessing(
        pd.read_csv("../../common_data/weather.csv")
    )

    # 静的データと動的データを結合
    odpt_df = pd.merge(static_df, dynamic_df, on="station_id", how="left")
    del_cols = ["UUnnamed: 0_x", "Unnamed: 0_y"]
    odpt_df.drop(del_cols, axis=1, inplace=True)

    df = pd.merge(
        odpt_df, weather_df,
        on=["month", "date", "hours"],
        how="left"
    )

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
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # ソート
    df = df.sort_values(["station_id", "datetime"])

    # 数値に変換
    df["num_bikes_available"] = pd.to_numeric(
        df["num_bikes_available"], errors="coerce")

    # 差分計算
    df["delta"] = df.groupby("station_id")["num_bikes_available"].diff()

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