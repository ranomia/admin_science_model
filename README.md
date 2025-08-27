# 行政事業レビュー科学技術事業分類モデル

このプロジェクトは、行政事業レビューデータを用いて、政府事業が科学技術関連事業であるかを自動分類するモデルを構築します。ModernBERTを使用した最新の自然言語処理技術により、高精度な分類を実現します。

## 📋 プロジェクト概要

### 目的
- 行政事業レビューデータから科学技術関連事業を自動分類
- ModernBERTの日本語行政文書への適用可能性の検証
- 行政事業レビューの効率化への貢献

### 対象データ
- **訓練データ**: `data/raw/train_deduplicated.csv` (13,078件)
- **評価データ**: `data/raw/test_deduplicated.csv` (2,350件)
- **目的変数**: `science_tech_decision` (該当/非該当の2値分類)

### 主要な特徴量
- 事業名 (`project_name`)
- 事業目的 (`project_objective`)
- 事業概要 (`project_summary`)
- 現在の課題 (`current_issues`)
- 所管省庁 (`responsible_ministry`)

## 🏗️ プロジェクト構造

```
admin_science_model/
├── data/
│   ├── raw/                    # 生データ
│   └── processed/              # 前処理済みデータ
├── src/
│   ├── config/
│   │   └── config.yaml         # 設定ファイル
│   ├── data/
│   │   ├── preprocessing.py    # データ前処理
│   │   └── dataset.py          # PyTorchデータセット
│   ├── models/
│   │   ├── baseline_models.py  # ベースラインモデル
│   │   └── modernbert_classifier.py  # ModernBERT分類器
│   ├── evaluation/
│   │   └── metrics.py          # 評価指標・可視化
│   └── main.py                 # メイン実行スクリプト
├── notebooks/
│   └── data_exploration.ipynb  # データ探索ノートブック
├── models/                     # 学習済みモデル保存先
├── results/                    # 実験結果保存先
├── requirements.txt            # 依存関係
└── README.md                   # このファイル
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# リポジトリをクローン
git clone <repository-url>
cd admin_science_model

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. GPU環境（オプション）

ModernBERTの訓練にはGPUの使用を強く推奨します。

```bash
# CUDA対応PyTorchのインストール（GPU使用時）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 実行方法

### Phase 1: データ探索・前処理・ベースライン構築

```bash
python src/main.py --phase 1
```

このフェーズでは以下を実行します：
- データの読み込みと基本統計
- 欠損値処理とテキスト正規化
- TF-IDF + 機械学習によるベースラインモデル構築
- ベースラインモデルの性能評価

### Phase 2: ModernBERTファインチューニング

```bash
python src/main.py --phase 2
```

このフェーズでは以下を実行します：
- ModernBERTモデルの初期化
- ファインチューニング実行
- ハイパーパラメータ最適化
- 学習済みモデルの保存

### Phase 3: モデル評価・比較分析

```bash
python src/main.py --phase 3
```

このフェーズでは以下を実行します：
- ベースラインとModernBERTの性能比較
- 詳細な評価指標の計算
- 結果の可視化
- 最終レポートの生成

### 全フェーズ一括実行

```bash
python src/main.py --phase all
```

### 個別設定での実行

```bash
# カスタム設定ファイルを使用
python src/main.py --config custom_config.yaml

# ログレベル変更
python src/main.py --log-level DEBUG
```

## 📈 データ探索

Jupyter Notebookを使用したインタラクティブなデータ探索：

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## ⚙️ 設定

`src/config/config.yaml`で各種パラメータを調整できます：

### 主要設定項目

```yaml
# モデル設定
model:
  name: "sbintuitions/modernbert-ja-130m"  # ModernBERTモデル
  max_length: 512                          # 最大トークン長
  dropout_rate: 0.1                        # ドロップアウト率

# 訓練設定
training:
  batch_size: 16                           # バッチサイズ
  learning_rate: 3e-5                      # 学習率
  num_epochs: 4                            # エポック数
  weight_decay: 0.01                       # 重み減衰
  early_stopping_patience: 2               # Early Stoppingの許容エポック

# 評価設定
evaluation:
  cv_folds: 5                              # クロスバリデーション分割数
  test_size: 0.2                           # 検証データ割合
```

## 📊 期待される結果

### 性能目標
- **主目標**: ROC-AUC 0.85以上
- **副目標**: ベースラインからの+5%以上の改善

### 出力ファイル
- `results/phase1_results.json`: Phase 1の詳細結果
- `results/phase2_results.json`: Phase 2の詳細結果
- `results/phase3_results.json`: Phase 3の詳細結果
- `results/all_results.json`: 全体の統合結果
- `models/best_model/`: 最良性能のModernBERTモデル
- `results/*.png`: 各種可視化図表

## 🔧 トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   ```bash
   # バッチサイズを削減
   # config.yamlのtraining.batch_sizeを8や4に変更
   ```

2. **CUDA out of memory**
   ```bash
   # より小さなmax_lengthを使用
   # config.yamlのmodel.max_lengthを256に変更
   ```

3. **ModernBERTモデルのダウンロードエラー**
   ```bash
   # インターネット接続を確認
   # Hugging Face Hubへのアクセスを確認
   ```

### ログの確認

```bash
# 詳細ログを確認
tail -f preprocessing.log

# エラーログを確認
python src/main.py --log-level DEBUG
```

## 📚 技術仕様

### 使用技術
- **フレームワーク**: PyTorch + Transformers
- **モデル**: ModernBERT (sbintuitions/modernbert-ja-130m)
- **ベースライン**: TF-IDF + ロジスティック回帰/LightGBM
- **評価**: 5-fold Cross Validation

### 必要環境
- Python 3.8以上
- GPU推奨（VRAM 8GB以上）
- RAM 16GB以上推奨

## 📄 ライセンス

このプロジェクトは研究・教育目的で作成されています。

## 🤝 貢献

バグ報告や改善提案は、IssueまたはPull Requestでお願いします。

## 📞 サポート

技術的な質問や問題がある場合は、以下の情報と共にお問い合わせください：
- 実行環境（OS、Python版、GPU情報）
- エラーメッセージ
- 実行したコマンド

---

**作成日**: 2025年1月  
**作成者**: AI開発チーム 