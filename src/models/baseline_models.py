"""
ベースラインモデルの実装.

このモジュールは以下の機能を提供します:
- TF-IDF特徴量抽出
- 従来の機械学習モデル（ランダムフォレスト等）
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from lightgbm import LGBMClassifier
from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution
import torch
from transformers import AutoTokenizer, AutoModel


class DataFrameLabelEncoder(BaseEstimator, TransformerMixin):
    """複数列に対してLabelEncoderを適用するトランスフォーマー."""

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = pd.DataFrame(X)
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna(''))
            self.encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        transformed = []
        for col in X.columns:
            le = self.encoders[col]
            transformed.append(
                X[col]
                .astype(str)
                .fillna('')
                .map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
                .astype(np.int32)
            )
        return np.vstack(transformed).T


class LaBSEEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """LaBSEモデルを用いて文の埋め込みを生成するトランスフォーマー."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Hugging Faceのモデルとトークナイザーを読み込む."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model", None)
        state.pop("tokenizer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = pd.Series(X).astype(str).tolist()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                token_emb = outputs.last_hidden_state
                attn_mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                sum_emb = torch.sum(token_emb * attn_mask, 1)
                sum_mask = torch.clamp(attn_mask.sum(1), min=1e-9)
                mean_emb = sum_emb / sum_mask
                embeddings.append(mean_emb.cpu().numpy())
        return np.vstack(embeddings)

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
        
    def evaluate(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series) -> Dict[str, float]:
        """
        モデルを評価する.

        Args:
            X: 特徴量（テキストまたは特徴量データフレーム）
            y: 目的変数

        Returns:
            評価指標の辞書
        """
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        # 予測確率
        y_prob = self.predict_proba(X)

        # ラベルをエンコード
        y_true = self._encode_labels(y)

        # ROC-AUCのみ計算
        return {
            'roc_auc': roc_auc_score(y_true, y_prob[:, 1])
        }
        
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


class TFIDFLightGBM(BaselineModel):
    """TF-IDF + LightGBMモデル."""

    def __init__(self, config: Dict):
        """初期化."""
        super().__init__(config)

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=config['evaluation']['random_state'],
            class_weight='balanced'
        )

        self.pipeline = None

    def _build_pipeline(self, X: pd.DataFrame) -> None:
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [
            col for col in X.select_dtypes(include=['object', 'category']).columns
            if col != 'combined_text'
        ]

        transformers = [('tfidf', self.vectorizer, 'combined_text')]
        if numeric_features:
            transformers.append(('num', 'passthrough', numeric_features))
        if categorical_features:
            transformers.append(
                ('cat', DataFrameLabelEncoder(), categorical_features)
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', self.model)
        ])

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series) -> None:
        self.logger.info("TF-IDF + LightGBMの訓練を開始します")

        if isinstance(X, pd.Series):
            X_df = X.to_frame('combined_text')
        else:
            X_df = X

        y_encoded = self._encode_labels(y)

        self._build_pipeline(X_df)
        param_distributions = {
            'classifier__n_estimators': IntDistribution(100, 1000),
            'classifier__learning_rate': FloatDistribution(1e-3, 0.3, log=True),
            'classifier__num_leaves': IntDistribution(20, 150),
            'classifier__max_depth': IntDistribution(3, 12),
            'classifier__min_child_samples': IntDistribution(1, 100),
            'classifier__subsample': FloatDistribution(0.6, 1.0),
            'classifier__colsample_bytree': FloatDistribution(0.6, 1.0),
            'classifier__reg_alpha': FloatDistribution(1e-8, 1.0, log=True),
            'classifier__reg_lambda': FloatDistribution(1e-8, 1.0, log=True),
            'classifier__min_split_gain': FloatDistribution(0.0, 0.2)
        }
        cv = StratifiedKFold(
            n_splits=self.config['evaluation']['cv_folds'],
            shuffle=True,
            random_state=self.config['evaluation']['random_state']
        )
        optuna_search = OptunaSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            cv=cv,
            scoring='roc_auc',
            n_trials=self.config['evaluation'].get('optuna_trials', 50),
            n_jobs=-1,
            random_state=self.config['evaluation']['random_state']
        )
        optuna_search.fit(X_df, y_encoded)
        self.pipeline = optuna_search.best_estimator_
        self.model = self.pipeline.named_steps['classifier']
        self.is_fitted = True

        self.logger.info("訓練が完了しました")

    def predict(self, X: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        X_df = X.to_frame('combined_text') if isinstance(X, pd.Series) else X
        return self.pipeline.predict(X_df)

    def predict_proba(self, X: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("モデルが訓練されていません")

        X_df = X.to_frame('combined_text') if isinstance(X, pd.Series) else X
        return self.pipeline.predict_proba(X_df)


class TFIDFLaBSELightGBM(TFIDFLightGBM):
    """TF-IDFとLaBSE埋め込みを組み合わせたLightGBMモデル."""

    def __init__(self, config: Dict, model_name: str = "sentence-transformers/LaBSE"):
        super().__init__(config)
        self.embedder = LaBSEEmbeddingTransformer(model_name=model_name)

    def _build_pipeline(self, X: pd.DataFrame) -> None:
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [
            col for col in X.select_dtypes(include=['object', 'category']).columns
            if col != 'combined_text'
        ]

        transformers = [
            ('tfidf', self.vectorizer, 'combined_text'),
            ('labse', self.embedder, 'combined_text')
        ]
        if numeric_features:
            transformers.append(('num', 'passthrough', numeric_features))
        if categorical_features:
            transformers.append(
                ('cat', DataFrameLabelEncoder(), categorical_features)
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', self.model)
        ])


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
        X_train: Union[pd.Series, pd.DataFrame],
        y_train: pd.Series,
        X_val: Union[pd.Series, pd.DataFrame],
        y_val: pd.Series
    ) -> pd.DataFrame:
        """
        複数のモデルを比較する.

        Args:
            models: モデルの辞書
            X_train: 訓練用特徴量（SeriesまたはDataFrame）
            y_train: 訓練用目的変数
            X_val: 検証用特徴量（SeriesまたはDataFrame）
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
        X: Union[pd.Series, pd.DataFrame],
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        クロスバリデーションを実行する.

        Args:
            model: 評価するモデル
            X: 特徴量（SeriesまたはDataFrame）
            y: 目的変数
            cv: 分割数
            
        Returns:
            各指標のスコア配列
        """
        self.logger.info(f"{cv}-fold クロスバリデーションを開始します")
        
        # ラベルをエンコード
        y_encoded = model._encode_labels(y)
        
        # クロスバリデーション実行（ROC-AUCのみ）
        scores = cross_val_score(
            model.pipeline, X, y_encoded,
            cv=cv, scoring='roc_auc',
            n_jobs=-1
        )
        return {'roc_auc': scores}
        
    def generate_detailed_report(
        self,
        model: BaselineModel,
        X_test: Union[pd.Series, pd.DataFrame],
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        詳細な評価レポートを生成する.
        
        Args:
            model: 評価するモデル
            X_test: テスト用特徴量（SeriesまたはDataFrame）
            y_test: テスト用目的変数
            model_name: モデル名
            
        Returns:
            詳細レポートの辞書
        """
        if not model.is_fitted:
            raise ValueError("モデルが訓練されていません")
            
        # 予測確率のみ取得
        y_prob = model.predict_proba(X_test)
        y_true = model._encode_labels(y_test)

        # ROC-AUCを計算
        metrics = model.evaluate(X_test, y_test)

        # 詳細レポート（ROC-AUCと確率のみ）
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': {
                'y_true': y_true,
                'y_prob': y_prob
            }
        }

        return report


def create_baseline_models(config: Dict) -> Dict[str, BaselineModel]:
    """
    ベースラインモデルを作成する.

    Args:
        config: 設定辞書

    Returns:
        ベースラインモデルの辞書
    """
    models = {
        'tfidf_lightgbm': TFIDFLightGBM(config),
        'tfidf_labse_lightgbm': TFIDFLaBSELightGBM(config)
    }

    return models
