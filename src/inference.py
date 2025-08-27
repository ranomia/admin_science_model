import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml

from data.preprocessing import DataPreprocessor, setup_logging
from models.modernbert_classifier import create_modernbert_model, ModernBERTTrainer
from data.dataset import create_data_loaders


def load_config(config_path: str) -> Dict:
    """設定ファイルを読み込む."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_inference(config_path: str, model_path: Path, output_path: Path) -> None:
    """ModernBERTモデルを用いてテストデータの予測を実行する."""
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config(config_path)

    # データ読み込みと前処理
    preprocessor = DataPreprocessor(config, logger)
    test_df = preprocessor.load_data(config["data"]["test_path"])
    test_processed = preprocessor.preprocess_pipeline(test_df)

    # モデルとトークナイザーの準備
    model, tokenizer = create_modernbert_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ModernBERTTrainer(model, tokenizer, config, device)
    trainer.load_model(model_path)

    # データローダーを作成し予測
    data_loaders = create_data_loaders(None, None, test_processed, tokenizer, config)
    preds, probs = trainer.predict(data_loaders["test"], return_probabilities=True)

    # ラベル名に変換
    label_mapping = {0: "非該当", 1: "該当"}
    predictions = [label_mapping[p] for p in preds]

    # 結果を保存
    output_df = test_df.copy()
    output_df["prediction"] = predictions
    output_df["probability_non"] = probs[:, 0]
    output_df["probability_yes"] = probs[:, 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df[["project_id", "prediction", "probability_yes"]].to_csv(output_path, index=False)
    logger.info(f"予測結果を保存しました: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="テストデータに対する推論スクリプト")
    parser.add_argument("--config", default="src/config/config.yaml", help="設定ファイルのパス")
    parser.add_argument("--model-path", default=None, help="読み込むモデルディレクトリ")
    parser.add_argument(
        "--output", default="results/test_predictions.csv", help="予測結果の保存先"
    )
    args = parser.parse_args()

    model_dir = (
        Path(args.model_path)
        if args.model_path is not None
        else Path(load_config(args.config)["output"]["model_dir"]) / "best_model"
    )
    output_path = Path(args.output)

    run_inference(args.config, model_dir, output_path)


if __name__ == "__main__":
    main()
