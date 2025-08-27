"""
ベースラインモデルの実装.

このモジュールは以下の機能を提供します:
- TF-IDF特徴量抽出
- 従来の機械学習モデル（ロジスティック回帰、ランダムフォレスト等）
- ベースライン性能の評価
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class BaselineModel:
    """ベースラインモデルのベースクラス."""
    
    def __init__(self, config: Dict):
        """
        初期化.
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.is_fitted = False
        
        # ラベルマッピング
        self.label_mapping = {'該当': 1, '非該当': 0}
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """
        ラベルを数値にエンコードする.
        
        Args:
            labels: ラベルのシリーズ
            
        Returns:
            エンコードされたラベル
        """
        return np.array([self.label_mapping.get(label, 0) for label in labels])
        
    def _decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        数値ラベルを文字列にデコードする.
        
        Args:
            encoded_labels: エンコードされたラベル
            
        Returns:
            デコードされたラベル
        """
        return [self.reverse_label_mapping.get(label, '非該当') for label in encoded_labels]
        
    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """
        モデルを訓練する.
        
        Args:
            X: 特徴量（テキスト）
            y: 目的変数
        """
        raise NotImplementedError
        
    def predict(self, X: pd.Series) -> np.ndarray:
        """
        予測を行う.
        
        Args:
            X: 特徴量（テキスト）
            
        Returns:
            予測結果
        """
        raise NotImplementedError
        
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        予測確率を計算する.
        
        Args:
            X: 特徴量（テキスト）
            
        Returns:
            予測確率
        """
        raise NotImplementedError
        
    def evaluate(self, X: pd.Series, y: pd.Series) -> Dict[str, float]:
        """
        モデルを評価する.
        
        Args:
            X: 特徴量（テキスト）
            y: 目的変数
            
        Returns:
            評価指標の辞書
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        # 予測
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        # ラベルをエンコード
        y_true = self._encode_labels(y)
        
        # 評価指標を計算
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_prob[:, 1])
        }
        
        return metrics
        
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        モデルを保存する.
        
        Args:
            filepath: 保存先のパス
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        model_data = {
            'pipeline': self.pipeline,
            'config': self.config,
            'label_mapping': self.label_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info(f"モデルを保存しました: {filepath}")
        
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        モデルを読み込む.
        
        Args:
            filepath: モデルファイルのパス
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.pipeline = model_data['pipeline']
        self.config = model_data['config']
        self.label_mapping = model_data['label_mapping']
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.is_fitted = True
        
        self.logger.info(f"モデルを読み込みました: {filepath}")


class TFIDFLogisticRegression(BaselineModel):
    """TF-IDF + ロジスティック回帰モデル."""
    
    def __init__(self, config: Dict):
        """初期化."""
        super().__init__(config)
        
        # TF-IDFベクトライザーの設定
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,  # 日本語のストップワードは別途対応
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # ロジスティック回帰の設定
        self.model = LogisticRegression(
            random_state=config['evaluation']['random_state'],
            max_iter=1000,
            class_weight='balanced'
        )
        
        # パイプライン作成
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """モデルを訓練する."""
        self.logger.info("TF-IDF + ロジスティック回帰の訓練を開始します")
        
        # ラベルをエンコード
        y_encoded = self._encode_labels(y)
        
        # 訓練
        self.pipeline.fit(X, y_encoded)
        self.is_fitted = True
        
        self.logger.info("訓練が完了しました")
        
    def predict(self, X: pd.Series) -> np.ndarray:
        """予測を行う."""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        return self.pipeline.predict(X)
        
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """予測確率を計算する."""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        return self.pipeline.predict_proba(X)


class TFIDFRandomForest(BaselineModel):
    """TF-IDF + ランダムフォレストモデル."""
    
    def __init__(self, config: Dict):
        """初期化."""
        super().__init__(config)
        
        # TF-IDFベクトライザーの設定
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # ランダムフォレスト用に特徴量数を削減
            ngram_range=(1, 2),
            stop_words=None,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # ランダムフォレストの設定
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=config['evaluation']['random_state'],
            class_weight='balanced',
            n_jobs=-1
        )
        
        # パイプライン作成
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """モデルを訓練する."""
        self.logger.info("TF-IDF + ランダムフォレストの訓練を開始します")
        
        # ラベルをエンコード
        y_encoded = self._encode_labels(y)
        
        # 訓練
        self.pipeline.fit(X, y_encoded)
        self.is_fitted = True
        
        self.logger.info("訓練が完了しました")
        
    def predict(self, X: pd.Series) -> np.ndarray:
        """予測を行う."""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        return self.pipeline.predict(X)
        
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """予測確率を計算する."""
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        return self.pipeline.predict_proba(X)
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        特徴量重要度を取得する.
        
        Args:
            feature_names: 特徴量名のリスト
            
        Returns:
            特徴量重要度のデータフレーム
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        # 特徴量重要度を取得
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        
        # 特徴量名を取得
        if feature_names is None:
            feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
            
        # データフレームに変換
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


class BaselineEvaluator:
    """ベースラインモデルの評価クラス."""
    
    def __init__(self, config: Dict):
        """
        初期化.
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def compare_models(
        self,
        models: Dict[str, BaselineModel],
        X_train: pd.Series,
        y_train: pd.Series,
        X_val: pd.Series,
        y_val: pd.Series
    ) -> pd.DataFrame:
        """
        複数のモデルを比較する.
        
        Args:
            models: モデルの辞書
            X_train: 訓練用特徴量
            y_train: 訓練用目的変数
            X_val: 検証用特徴量
            y_val: 検証用目的変数
            
        Returns:
            比較結果のデータフレーム
        """
        results = []
        
        for model_name, model in models.items():
            self.logger.info(f"{model_name}の評価を開始します")
            
            # 訓練
            model.fit(X_train, y_train)
            
            # 評価
            train_metrics = model.evaluate(X_train, y_train)
            val_metrics = model.evaluate(X_val, y_val)
            
            # 結果を記録
            for metric_name, train_score in train_metrics.items():
                val_score = val_metrics[metric_name]
                results.append({
                    'model': model_name,
                    'metric': metric_name,
                    'train_score': train_score,
                    'val_score': val_score,
                    'overfitting': train_score - val_score
                })
                
        return pd.DataFrame(results)
        
    def cross_validate_model(
        self,
        model: BaselineModel,
        X: pd.Series,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        クロスバリデーションを実行する.
        
        Args:
            model: 評価するモデル
            X: 特徴量
            y: 目的変数
            cv: 分割数
            
        Returns:
            各指標のスコア配列
        """
        self.logger.info(f"{cv}-fold クロスバリデーションを開始します")
        
        # ラベルをエンコード
        y_encoded = model._encode_labels(y)
        
        # クロスバリデーション実行
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = {}
        
        for score in scoring:
            scores = cross_val_score(
                model.pipeline, X, y_encoded,
                cv=cv, scoring=score,
                n_jobs=-1
            )
            cv_results[score] = scores
            
        return cv_results
        
    def generate_detailed_report(
        self,
        model: BaselineModel,
        X_test: pd.Series,
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        詳細な評価レポートを生成する.
        
        Args:
            model: 評価するモデル
            X_test: テスト用特徴量
            y_test: テスト用目的変数
            model_name: モデル名
            
        Returns:
            詳細レポートの辞書
        """
        if not model.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        # 予測
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        y_true = model._encode_labels(y_test)
        
        # 基本的な評価指標
        metrics = model.evaluate(X_test, y_test)
        
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        
        # 分類レポート
        class_report = classification_report(
            y_true, y_pred,
            target_names=['非該当', '該当'],
            output_dict=True
        )
        
        # 詳細レポート
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        }
        
        return report


def create_baseline_models(config: Dict) -> Dict[str, BaselineModel]:
    """
    ベースラインモデルを作成する.

    現時点では精度の低かった手法を除外し、
    TF-IDF + ロジスティック回帰のみを提供する。

    Args:
        config: 設定辞書

    Returns:
        ベースラインモデルの辞書
    """
    models = {
        'tfidf_logistic': TFIDFLogisticRegression(config)
    }

    return models
