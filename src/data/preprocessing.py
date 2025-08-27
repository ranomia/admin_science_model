"""
行政事業レビューデータの前処理モジュール.

このモジュールは以下の機能を提供します:
- データの読み込みと基本統計
- 欠損値処理
- テキスト正規化
- 特徴量結合
- データ分割
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """行政事業レビューデータの前処理クラス."""
    
    def __init__(self, config: Dict):
        """
        初期化.
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        CSVファイルからデータを読み込む.
        
        Args:
            file_path: CSVファイルのパス
            
        Returns:
            読み込んだデータフレーム
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"データを読み込みました: {file_path}")
            self.logger.info(f"データ形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}")
            raise
            
    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """
        データの基本統計を分析する.
        
        Args:
            df: 分析対象のデータフレーム
            
        Returns:
            分析結果の辞書
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_rates': (df.isnull().sum() / len(df)).to_dict()
        }
        
        # 目的変数の分布
        target_col = self.config['data']['target_column']
        if target_col in df.columns:
            analysis['target_distribution'] = df[target_col].value_counts().to_dict()
            analysis['target_ratio'] = df[target_col].value_counts(normalize=True).to_dict()
        
        # テキスト列の長さ統計
        text_columns = self.config['data']['text_columns']
        text_stats = {}
        for col in text_columns:
            if col in df.columns:
                text_lengths = df[col].astype(str).str.len()
                text_stats[col] = {
                    'mean': text_lengths.mean(),
                    'median': text_lengths.median(),
                    'max': text_lengths.max(),
                    'min': text_lengths.min(),
                    'std': text_lengths.std()
                }
        analysis['text_statistics'] = text_stats
        
        return analysis
        
    def clean_text(self, text: str) -> str:
        """
        テキストを正規化する.
        
        Args:
            text: 元のテキスト
            
        Returns:
            正規化されたテキスト
        """
        if pd.isna(text) or text == '':
            return ''
            
        # 文字列に変換
        text = str(text)
        
        # 改行文字を空白に置換
        text = re.sub(r'\r\n|\r|\n', ' ', text)
        
        # 連続する空白を単一の空白に置換
        text = re.sub(r'\s+', ' ', text)
        
        # 前後の空白を除去
        text = text.strip()
        
        return text
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を処理する.
        
        Args:
            df: 処理対象のデータフレーム
            
        Returns:
            欠損値処理後のデータフレーム
        """
        df_processed = df.copy()
        
        # テキスト列の欠損値を空文字列で埋める
        text_columns = self.config['data']['text_columns']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('')
                
        # 数値列の欠損値を0で埋める
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_processed[col] = df_processed[col].fillna(0)
            
        self.logger.info("欠損値処理を完了しました")
        return df_processed
        
    def combine_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テキスト特徴量を結合する.
        
        Args:
            df: 処理対象のデータフレーム
            
        Returns:
            テキスト特徴量結合後のデータフレーム
        """
        df_processed = df.copy()
        
        # テキスト列をクリーニング
        text_columns = self.config['data']['text_columns']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(self.clean_text)
        
        # テキスト特徴量を結合
        text_parts = []
        for col in text_columns:
            if col in df_processed.columns:
                text_parts.append(df_processed[col])
        
        # 結合テキストを作成（空でないものだけを結合）
        combined_texts = []
        for idx in range(len(df_processed)):
            parts = [str(text_parts[i].iloc[idx]) for i in range(len(text_parts)) 
                    if str(text_parts[i].iloc[idx]).strip() != '']
            combined_text = ' '.join(parts)
            combined_texts.append(combined_text)
            
        df_processed['combined_text'] = combined_texts
        
        self.logger.info("テキスト特徴量の結合を完了しました")
        return df_processed
        
    def truncate_text(self, text: str, max_length: int = 512) -> str:
        """
        テキストを指定した最大長で切り詰める.
        
        Args:
            text: 元のテキスト
            max_length: 最大文字数
            
        Returns:
            切り詰められたテキスト
        """
        if len(text) <= max_length:
            return text
        return text[:max_length]
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        機械学習用の特徴量を準備する.
        
        Args:
            df: 処理対象のデータフレーム
            
        Returns:
            特徴量準備後のデータフレーム
        """
        df_processed = df.copy()
        
        # 禁止列を除去
        forbidden_columns = self.config['data']['forbidden_columns']
        for col in forbidden_columns:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                self.logger.info(f"禁止列を除去しました: {col}")
        
        # テキスト長特徴量を作成
        text_columns = self.config['data']['text_columns']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[f"{col}_length"] = df_processed[col].astype(str).str.len()

        if 'combined_text' in df_processed.columns:
            df_processed['combined_text'] = df_processed['combined_text'].astype(str)
            df_processed['combined_text_length'] = df_processed['combined_text'].str.len()

        # テキスト長を制限
        max_length = int(self.config['model']['max_length'])
        if 'combined_text' in df_processed.columns:
            df_processed['combined_text'] = df_processed['combined_text'].apply(
                lambda x: self.truncate_text(x, max_length)
            )

        # 元の個別テキスト列は除去
        for col in text_columns:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])

        return df_processed
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データを訓練用と検証用に分割する.
        
        Args:
            df: 分割対象のデータフレーム
            
        Returns:
            (訓練データ, 検証データ)のタプル
        """
        target_col = self.config['data']['target_column']
        test_size = float(self.config['evaluation']['test_size'])
        random_state = int(self.config['evaluation']['random_state'])
        stratify = self.config['evaluation']['stratify']
        
        if stratify and target_col in df.columns:
            stratify_col = df[target_col]
        else:
            stratify_col = None
            
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        self.logger.info(f"データ分割完了 - 訓練: {len(train_df)}, 検証: {len(val_df)}")
        return train_df, val_df
        
    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        前処理パイプライン全体を実行する.
        
        Args:
            df: 処理対象のデータフレーム
            
        Returns:
            前処理完了後のデータフレーム
        """
        self.logger.info("前処理パイプラインを開始します")
        
        # 1. 欠損値処理
        df = self.handle_missing_values(df)
        
        # 2. テキスト特徴量結合
        df = self.combine_text_features(df)
        
        # 3. 特徴量準備
        df = self.prepare_features(df)
        
        self.logger.info("前処理パイプラインを完了しました")
        return df
        
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        前処理済みデータを保存する.
        
        Args:
            df: 保存するデータフレーム
            filename: ファイル名
        """
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)
        self.logger.info(f"前処理済みデータを保存しました: {output_path}")


def setup_logging(level: str = 'INFO') -> None:
    """
    ログ設定をセットアップする.
    
    Args:
        level: ログレベル
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing.log')
        ]
    ) 