"""
評価指標の計算と可視化.

このモジュールは以下の機能を提供します:
- 各種評価指標の計算
- 混同行列の可視化
- ROC曲線・PR曲線の作成
- 結果の可視化
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc, precision_recall_curve,
    roc_auc_score, roc_curve
)

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


class MetricsCalculator:
    """評価指標計算クラス."""
    
    def __init__(self):
        """初期化."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        ROC-AUCのみを計算する.

        Args:
            y_true: 真のラベル
            y_pred: 未使用（互換性のため）
            y_prob: 予測確率
            average: 未使用

        Returns:
            {'roc_auc': スコア} の辞書
        """
        metrics: Dict[str, float] = {}

        if y_prob is not None and len(np.unique(y_true)) == 2:
            if y_prob.ndim > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_positive)

        return metrics


class MetricsVisualizer:
    """評価指標可視化クラス."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        初期化.
        
        Args:
            figsize: 図のサイズ
        """
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = False,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        混同行列を可視化する.
        
        Args:
            cm: 混同行列
            class_names: クラス名のリスト
            title: タイトル
            normalize: 正規化するかどうか
            save_path: 保存先のパス
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"混同行列を保存しました: {save_path}")
            
        return fig
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        ROC曲線を描画する.
        
        Args:
            y_true: 真のラベル
            y_prob: 予測確率
            title: タイトル
            save_path: 保存先のパス
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # ROC曲線を計算
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # 描画
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC曲線を保存しました: {save_path}")
            
        return fig
        
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Precision-Recall曲線を描画する.
        
        Args:
            y_true: 真のラベル
            y_prob: 予測確率
            title: タイトル
            save_path: 保存先のパス
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # PR曲線を計算
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # 描画
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR曲線を保存しました: {save_path}")
            
        return fig
        
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        訓練履歴を可視化する.
        
        Args:
            history: 訓練履歴の辞書
            title: タイトル
            save_path: 保存先のパス
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['train_loss', 'val_loss', 'val_roc_auc']
        titles = ['Training Loss', 'Validation Loss', 'Validation ROC-AUC']

        for ax, metric, metric_title in zip(axes, metrics, titles):
            if metric in history and history[metric]:
                ax.plot(history[metric], label=metric_title)
                ax.set_title(metric_title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"訓練履歴を保存しました: {save_path}")
            
        return fig
        
    def plot_metrics_comparison(
        self,
        results_df: pd.DataFrame,
        title: str = "Model Comparison",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        モデル比較結果を可視化する.
        
        Args:
            results_df: 比較結果のデータフレーム
            title: タイトル
            save_path: 保存先のパス
            
        Returns:
            matplotlib図オブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # メトリック別にグループ化
        metrics = results_df['metric'].unique()
        models = results_df['model'].unique()
        
        # バープロット用のデータを準備
        pivot_df = results_df.pivot(index='model', columns='metric', values='val_score')
        
        # バープロット
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"モデル比較を保存しました: {save_path}")
            
        return fig


class ModelEvaluator:
    """モデル評価統合クラス."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初期化.
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self.metrics_calc = MetricsCalculator()
        self.visualizer = MetricsVisualizer()
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
    ) -> Dict:
        """
        モデルの評価を実行する（ROC-AUCのみ）.

        Args:
            y_true: 真のラベル
            y_prob: 予測確率
            model_name: モデル名

        Returns:
            評価結果の辞書
        """
        self.logger.info(f"{model_name}の評価を開始します")

        # 指標を計算
        metrics = self.metrics_calc.calculate_all_metrics(y_true, y_prob=y_prob)

        # 結果をまとめる
        results = {
            'model_name': model_name,
            'metrics': metrics,
        }

        # 可視化
        if self.output_dir:
            self._create_visualizations(y_true, y_prob, results)

        self.logger.info(f"{model_name}の評価が完了しました")
        return results
        
    def _create_visualizations(
        self,
        y_true: np.ndarray,
        y_prob: Optional[np.ndarray],
        results: Dict
    ) -> None:
        """
        ROC曲線などの可視化を作成する.

        Args:
            y_true: 真のラベル
            y_prob: 予測確率
            results: 評価結果
        """
        model_name = results['model_name']

        if y_prob is not None and len(np.unique(y_true)) == 2:
            if y_prob.ndim > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob

            roc_path = self.output_dir / f"{model_name}_roc_curve.png"
            self.visualizer.plot_roc_curve(
                y_true, y_prob_positive,
                title=f"{model_name} - ROC Curve",
                save_path=roc_path
            )

            pr_path = self.output_dir / f"{model_name}_pr_curve.png"
            self.visualizer.plot_precision_recall_curve(
                y_true, y_prob_positive,
                title=f"{model_name} - Precision-Recall Curve",
                save_path=pr_path
            )
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json") -> None:
        """
        評価結果を保存する.
        
        Args:
            results: 評価結果
            filename: ファイル名
        """
        if not self.output_dir:
            return
            
        import json
        
        # NumPy配列をリストに変換
        serializable_results = self._make_serializable(results)
        
        save_path = self.output_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"評価結果を保存しました: {save_path}")
        
    def _make_serializable(self, obj):
        """オブジェクトをシリアライズ可能に変換する."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj 