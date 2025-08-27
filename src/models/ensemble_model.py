"""モデル統合のためのアンサンブル実装.

このモジュールはベースラインモデルとModernBERTモデルを
統合して予測するためのアンサンブルクラスを提供します。
"""

from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .baseline_models import BaselineModel
from .modernbert_classifier import ModernBERTTrainer
from evaluation.metrics import MetricsCalculator


class EnsembleModel:
    """ベースラインモデルとModernBERTを統合するアンサンブルモデル."""

    def __init__(
        self,
        baseline_model: BaselineModel,
        bert_trainer: ModernBERTTrainer,
        weights: Tuple[float, float] = (0.5, 0.5),
    ) -> None:
        """初期化.

        Args:
            baseline_model: TF-IDFなどのベースラインモデル
            bert_trainer: ModernBERTのトレーナー
            weights: 各モデルに割り当てる重み (baseline, bert)
        """
        self.baseline_model = baseline_model
        self.bert_trainer = bert_trainer
        self.weights = weights
        self.metrics = MetricsCalculator()

    def predict(
        self,
        X_text: pd.Series,
        bert_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """アンサンブル予測を行う.

        Args:
            X_text: ベースラインモデル用のテキスト入力
            bert_loader: ModernBERT用のデータローダー

        Returns:
            予測ラベルと予測確率
        """
        baseline_probs = self.baseline_model.predict_proba(X_text)
        _, bert_probs = self.bert_trainer.predict(bert_loader, return_probabilities=True)

        ensemble_probs = (
            self.weights[0] * baseline_probs + self.weights[1] * bert_probs
        )
        preds = np.argmax(ensemble_probs, axis=1)
        return preds, ensemble_probs

    def evaluate(
        self,
        X_text: pd.Series,
        bert_loader: DataLoader,
        y_true: pd.Series,
    ) -> dict:
        """アンサンブルモデルを評価する.

        Args:
            X_text: テキスト入力
            bert_loader: データローダー
            y_true: 正解ラベル

        Returns:
            評価指標の辞書
        """
        y_encoded = self.baseline_model._encode_labels(y_true)
        _, probs = self.predict(X_text, bert_loader)
        return self.metrics.calculate_all_metrics(y_encoded, y_prob=probs)

