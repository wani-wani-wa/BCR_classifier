# 🧬 BCR Classifier

`BCR_classifier` は、B細胞受容体（BCR）配列を分類する PyTorch Lightning ベースのモデルです。Facebook のタンパク質言語モデル `ESM2` をベースに、SARS-CoV-2 や HIV-1 などのターゲット分類を行います。

---

## 🚀 特徴

- `facebook/esm2_t6_8M_UR50D` によるトークナイズ
- LightningModule / LightningDataModule による構造化されたモデル設計
- TensorBoard による学習可視化
- pytest によるユニットテスト
- GitHub Actions を使った CI 自動テスト

---

## 📂 データ準備

学習に使用するデータは、以下の2つのノートブックを順に実行して事前処理する必要があります：

1. [`bcr_preprocessing.ipynb`](https://github.com/naity/protein-transformer/blob/main/notebooks/bcr_preprocessing.ipynb)
   → BCR 配列の収集・前処理（フィルタリングなど）

2. [`prepare_dataset.ipynb`](https://github.com/wani-wani-wa/BCR_classifier/blob/main/scripts/prepare_dataset.ipynb)
   → クラスラベルのエンコード・train/val/test への分割、`.parquet` ファイルとして保存

保存先：
```bash
data_dir/
├── bcr_train.parquet
├── bcr_val.parquet
└── bcr_test.parquet
```
---

## 🚀 モデル学習

```bash
python protein_transformer/train.py \
    --data_dir ./data_dir \
    --max_epochs 10 \
    --batch_size 32 \
    --gpus 1
```
- `--data_dir`：データセットのディレクトリ, デフォルトは `./data_dir`
- `--max_epochs`：学習エポック数, デフォルトは 10
- `--batch_size`：バッチサイズ, デフォルトは 32
- `--gpus`：使用するGPUの数, デフォルトは 1

## TensorBoard での可視化
```bash
tensorboard --logdir lightning_logs/
```
---
## 🏗️ ディレクトリ構成
```bash
BCR_classifier/
├── scripts/                     # 学習・推論用スクリプト
│   ├── train.py
│   └── prepare_dataset.ipynb
├── protein_transformer/        # データ・モデルモジュール
│   ├── data/
│   │   ├── dataset.py
│   │   └── datamodule.py
│   ├── model/
│   │   └── lightning_module.py
│   └── __init__.py
├── tests/                      # 単体テスト（pytest）
│   ├── test_dataset.py
│   ├── test_datamodule.py
│   └── test_model.py
├── data_dir/                   # 前処理済みデータ保存先
│   ├── bcr_train.parquet
│   ├── bcr_val.parquet
│   └── bcr_test.parquet
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions のCI定義
```
---
## 🧪 テスト

```bash
pytest -s tests/
```
- `tests/test_dataset.py`：データセットのユニットテスト
- `tests/test_datamodule.py`：データモジュールのユニットテスト
- `tests/test_model.py`：モデルのユニットテスト

___
## ⚙️ 開発環境構築（venv を使用）

```bash
# リポジトリをクローン
git clone https://github.com/wani-wani-wa/BCR_classifier.git
cd BCR_classifier

# uv のインストール（初回のみ）
curl -Ls https://astral.sh/uv/install.sh | sh

# 依存パッケージをインストール
uv pip install -r requirements.txt

# BCR_Classifierパッケージを開発モードでインストール
uv pip install -e .

# テストを実行
pytest -s tests/
```
___
## 🧼 CI（GitHub Actions）
.github/workflows/ci.yml により、main 以外のブランチも含めて push / pull request 時に以下を自動で実行します：

Python のセットアップ（3.11）

requirements.txt の依存関係インストール

単体テスト（pytest）実行

モジュールの import 確認