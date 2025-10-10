# OJT-share-cycle
OJTのシェアサイクル需要予測・最適化のリポジトリです。

## 概要
OJT用のシェアサイクル需要予測・最適化プロジェクトを管理するためのリポジトリ。

## フォルダの説明
- common_data: 常用データの保管
- forecasting: 需要予測コードの保管
- optimization: 最適化コードの保管
- log_setter: logの管理システムを構築するコードの保管
- utils: 常用関数コードの保管

## リポジトリ構成
```
.
├── common_data/
│   ├── station_information.json
│   ├── station_status.json
│   └── weather.csv
├── config →yaml/json/
│   ├── default
│   └── experiment →defaultから変更したもの
├── forecasting/
│   └── modeling.py
├── log_setter
├── optimization
├── utils/
│   ├── preprocessing.py →データロード・前処理用関数
│   └── result.py →評価・図
├── venv
├── .gitignore
├── README.md
└── requirements.txt
```

## ファイルの説明
- .gitignore: 不要なコミットメントの防止ファイル
- requirements.txt: pythonライブラリを管理するファイル

## ローカル環境のセットアップ
仮想環境を構築する
```
python3 -m venv venv
source ./venv/bin/activate
```

pip3を使用する場合, リポジトリのターミナル上で以下のコマンドを実行する
```
pip3 install -U pip
pip3 install -r requirements.txt
```

## 実行

## Commitルール
Commit の際は以下のルールに合わせて種類ごとにする.

🎉 初めてのコミット (Initial Commit)  
🔖 バージョンタグ (Version Tag)  
✨ 新機能 (New Feature)  
🐛 バグ修正 (Bugfix)  
♻️ リファクタリング (Refactoring)  
📚 ドキュメント (Documentation)  
🎨 デザインUI/UX (Accessibility)  
🐎 パフォーマンス (Performance)  
🔧 ツール (Tooling)  
🚨 テスト (Tests)  
💩 非推奨追加 (Deprecation)  
🗑️ 削除 (Removal)  
🚧 WIP (Work In Progress)  