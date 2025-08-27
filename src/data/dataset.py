"""
PyTorchデータセットクラスの実装.

このモジュールは以下の機能を提供します:
- 行政事業レビューデータ用のカスタムDatasetクラス
- トークナイザーとの統合
- バッチ処理のためのコレート関数
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class AdminReviewDataset(Dataset):
    """行政事業レビューデータ用のPyTorchデータセットクラス."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        text_column: str = 'combined_text',
        target_column: str = 'science_tech_decision',
        max_length: int = 512,
        include_targets: bool = True
    ):
        """
        初期化.
        
        Args:
            df: データフレーム
            tokenizer: トークナイザー
            text_column: テキスト列名
            target_column: 目的変数列名
            max_length: 最大トークン長
            include_targets: 目的変数を含むかどうか
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.target_column = target_column
        self.max_length = max_length
        self.include_targets = include_targets
        self.logger = logging.getLogger(__name__)
        
        # ラベルマッピング
        self.label_mapping = {'該当': 1, '非該当': 0}
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        self.logger.info(f"データセットを初期化しました: {len(self.df)}件")
        
    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        指定されたインデックスのデータを取得する.
        
        Args:
            idx: データのインデックス
            
        Returns:
            トークナイズされたデータの辞書
        """
        # テキストデータを取得
        text = str(self.df.iloc[idx][self.text_column])
        
        # トークナイズ
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 結果を辞書に格納
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # トークンタイプIDがある場合は追加
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids'].squeeze()
        
        # 目的変数を含む場合
        if self.include_targets and self.target_column in self.df.columns:
            label_str = self.df.iloc[idx][self.target_column]
            label = self.label_mapping.get(label_str, 0)
            result['labels'] = torch.tensor(label, dtype=torch.long)
            
        return result
        
    def get_label_distribution(self) -> Dict[str, int]:
        """
        ラベルの分布を取得する.
        
        Returns:
            ラベル分布の辞書
        """
        if not self.include_targets or self.target_column not in self.df.columns:
            return {}
            
        return self.df[self.target_column].value_counts().to_dict()
        
    def get_text_lengths(self) -> List[int]:
        """
        テキストの長さリストを取得する.
        
        Returns:
            テキスト長のリスト
        """
        return [len(str(self.df.iloc[i][self.text_column])) for i in range(len(self.df))]


class DataCollator:
    """バッチ処理用のデータコレーター."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        初期化.
        
        Args:
            tokenizer: トークナイザー
        """
        self.tokenizer = tokenizer
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        バッチを処理する.
        
        Args:
            batch: バッチデータのリスト
            
        Returns:
            処理されたバッチデータ
        """
        # 各キーごとにテンソルをスタック
        result = {}
        
        # 必須キー
        for key in ['input_ids', 'attention_mask']:
            if key in batch[0]:
                result[key] = torch.stack([item[key] for item in batch])
                
        # オプショナルキー
        optional_keys = ['token_type_ids', 'labels']
        for key in optional_keys:
            if key in batch[0]:
                result[key] = torch.stack([item[key] for item in batch])
                
        return result


def create_data_loaders(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    tokenizer: PreTrainedTokenizer,
    config: Dict,
    text_column: str = 'combined_text',
    target_column: str = 'science_tech_decision'
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    データローダーを作成する.
    
    Args:
        train_df: 訓練データ
        val_df: 検証データ
        test_df: テストデータ
        tokenizer: トークナイザー
        config: 設定辞書
        text_column: テキスト列名
        target_column: 目的変数列名
        
    Returns:
        データローダーの辞書
    """
    logger = logging.getLogger(__name__)
    
    # 設定値を取得
    batch_size = int(config['training']['batch_size'])
    max_length = int(config['model']['max_length'])
    
    # データコレーターを作成
    data_collator = DataCollator(tokenizer)
    
    # データローダーの辞書を初期化
    data_loaders = {}
    
    # 訓練データローダー
    if train_df is not None:
        train_dataset = AdminReviewDataset(
            df=train_df,
            tokenizer=tokenizer,
            text_column=text_column,
            target_column=target_column,
            max_length=max_length,
            include_targets=True
        )
        
        data_loaders['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=0,  # Windowsでは0に設定
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # ラベル分布をログ出力（訓練データがある場合のみ）
        train_dist = train_dataset.get_label_distribution()
        logger.info(f"訓練データのラベル分布: {train_dist}")
    
    # 検証データローダー
    if val_df is not None:
        val_dataset = AdminReviewDataset(
            df=val_df,
            tokenizer=tokenizer,
            text_column=text_column,
            target_column=target_column,
            max_length=max_length,
            include_targets=True
        )
        
        data_loaders['val'] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # テストデータローダー
    if test_df is not None:
        test_dataset = AdminReviewDataset(
            df=test_df,
            tokenizer=tokenizer,
            text_column=text_column,
            target_column=target_column,
            max_length=max_length,
            include_targets=True if target_column in test_df.columns else False
        )
        
        data_loaders['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    logger.info(f"データローダーを作成しました: {list(data_loaders.keys())}")
    
    return data_loaders


def analyze_dataset_statistics(dataset: AdminReviewDataset) -> Dict:
    """
    データセットの統計情報を分析する.
    
    Args:
        dataset: 分析対象のデータセット
        
    Returns:
        統計情報の辞書
    """
    stats = {
        'size': len(dataset),
        'label_distribution': dataset.get_label_distribution(),
        'text_lengths': dataset.get_text_lengths()
    }
    
    # テキスト長の統計
    text_lengths = stats['text_lengths']
    if text_lengths:
        import numpy as np
        stats['text_length_stats'] = {
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'percentiles': {
                '25': np.percentile(text_lengths, 25),
                '75': np.percentile(text_lengths, 75),
                '90': np.percentile(text_lengths, 90),
                '95': np.percentile(text_lengths, 95)
            }
        }
    
    return stats 