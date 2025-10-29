import json

def json_data_load(json_path):
    """jsonファイルを読み込む関数"""

    with open(json_path, "r", encoding="utf-8") as f:
        df = json.load(f)

    df = df["data"]["stations"]

    return df