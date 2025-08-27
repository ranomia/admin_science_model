"""
ModernBERT分類器の実装.

このモジュールは以下の機能を提供します:
- ModernBERTを使用した2値分類器
- ファインチューニング機能
- 推論機能
- モデルの保存・読み込み
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np


class ModernBERTClassifier(nn.Module):
    """ModernBERTを使用した分類器."""
    
    def __init__(
        self,
        model_name: str = "sbintuitions/modernbert-ja-130m",
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        hidden_size: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        初期化.
        
        Args:
            model_name: 事前学習済みモデル名
            num_labels: ラベル数
            dropout_rate: ドロップアウト率
            hidden_size: 隠れ層のサイズ
        """
        super().__init__()
        
        # 設定を読み込み
        self.config = AutoConfig.from_pretrained(model_name)
        self.num_labels = num_labels
        
        # ModernBERTモデルを読み込み
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 分類器のヘッド
        if hidden_size is None:
            hidden_size = self.config.hidden_size
            
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # クラス重みを保存（クラス不均衡対策）
        self.class_weights = class_weights

        # 重みの初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みを初期化する."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        順伝播.
        
        Args:
            input_ids: 入力トークンID
            attention_mask: アテンションマスク
            token_type_ids: トークンタイプID
            labels: ラベル（訓練時のみ）
            
        Returns:
            出力の辞書
        """
        # BERTの出力を取得
        bert_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if token_type_ids is not None:
            bert_inputs['token_type_ids'] = token_type_ids
            
        outputs = self.bert(**bert_inputs)
        
        # [CLS]トークンの隠れ状態を取得
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # ドロップアウトを適用
        pooled_output = self.dropout(pooled_output)
        
        # 分類スコアを計算
        logits = self.classifier(pooled_output)
        
        # 結果を辞書に格納
        result = {'logits': logits}
        
        # 損失を計算（訓練時）
        if labels is not None:
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            result['loss'] = loss
            
        return result


class ModernBERTTrainer:
    """ModernBERT分類器の訓練クラス."""
    
    def __init__(
        self,
        model: ModernBERTClassifier,
        tokenizer: AutoTokenizer,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        初期化.
        
        Args:
            model: 分類器モデル
            tokenizer: トークナイザー
            config: 設定辞書
            device: デバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # モデルをデバイスに移動
        self.model.to(self.device)
        
        # 最適化器とスケジューラー
        self.optimizer = None
        self.scheduler = None
        
        # 訓練履歴
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_roc_auc': []
        }
        
    def setup_optimizer(self, num_training_steps: int) -> None:
        """
        最適化器とスケジューラーをセットアップする.
        
        Args:
            num_training_steps: 総訓練ステップ数
        """
        # 最適化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # スケジューラー
        num_warmup_steps = int(num_training_steps * float(self.config['training']['warmup_ratio']))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"最適化器をセットアップしました - 総ステップ数: {num_training_steps}, ウォームアップ: {num_warmup_steps}")
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        1エポック分の訓練を実行する.
        
        Args:
            train_loader: 訓練データローダー
            
        Returns:
            平均損失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # バッチをデバイスに移動
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 勾配をゼロにリセット
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            self.optimizer.step()
            self.scheduler.step()
            
            # 損失を累積
            total_loss += loss.item()
            num_batches += 1
            
            # プログレスバーを更新
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / num_batches
        
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        評価を実行する.
        
        Args:
            eval_loader: 評価データローダー
            
        Returns:
            評価指標の辞書
        """
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # バッチをデバイスに移動
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 順伝播
                outputs = self.model(**batch)
                
                # 損失を累積
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                
                # 予測確率を取得
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)

                if 'labels' in batch:
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
        
        # 評価指標を計算
        metrics = {}
        if all_labels:
            from sklearn.metrics import roc_auc_score

            metrics = {
                'loss': total_loss / len(eval_loader),
                'roc_auc': roc_auc_score(all_labels, all_probs)
            }
        
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """
        モデルを訓練する.
        
        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            save_dir: モデル保存ディレクトリ
            
        Returns:
            訓練履歴
        """
        if num_epochs is None:
            num_epochs = int(self.config['training']['num_epochs'])
            
        # 最適化器をセットアップ
        num_training_steps = len(train_loader) * num_epochs
        self.setup_optimizer(num_training_steps)
        
        self.logger.info(f"訓練を開始します - エポック数: {num_epochs}")
        
        best_roc_auc = 0.0

        patience = int(self.config['training'].get('early_stopping_patience', 0))
        patience_counter = 0

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 訓練
            train_loss = self.train_epoch(train_loader)
            self.train_history['train_loss'].append(train_loss)
            
            # 検証
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)

                self.train_history['val_loss'].append(val_metrics.get('loss', 0.0))
                self.train_history['val_roc_auc'].append(val_metrics.get('roc_auc', 0.0))

                self.logger.info(f"Train Loss: {train_loss:.4f}")
                self.logger.info(f"Val Loss: {val_metrics.get('loss', 0.0):.4f}")
                self.logger.info(f"Val ROC-AUC: {val_metrics.get('roc_auc', 0.0):.4f}")

                # ベストモデルを保存
                current_roc_auc = val_metrics.get('roc_auc', 0.0)
                if current_roc_auc > best_roc_auc:
                    best_roc_auc = current_roc_auc
                    patience_counter = 0
                    if save_dir is not None:
                        self.save_model(save_dir / 'best_model')
                        self.logger.info(f"ベストモデルを保存しました (ROC-AUC: {best_roc_auc:.4f})")

                else:
                    patience_counter += 1
                    if patience > 0 and patience_counter >= patience:
                        self.logger.info("Early stopping の条件を満たしたため訓練を終了します")
                        break
        
        self.logger.info("訓練が完了しました")
        return self.train_history
        
    def predict(
        self,
        data_loader: DataLoader,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        予測を実行する.
        
        Args:
            data_loader: データローダー
            return_probabilities: 確率も返すかどうか
            
        Returns:
            予測結果（確率も含む場合はタプル）
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                # バッチをデバイスに移動
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                # 順伝播
                outputs = self.model(**batch)
                logits = outputs['logits']
                
                # 予測と確率を計算
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        predictions = np.array(all_preds)
        probabilities = np.array(all_probs)
        
        if return_probabilities:
            return predictions, probabilities
        return predictions
        
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        モデルを保存する.
        
        Args:
            save_path: 保存先のパス
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # モデルの状態を保存
        torch.save(self.model.state_dict(), save_path / 'model.pth')
        
        # トークナイザーを保存
        self.tokenizer.save_pretrained(save_path)
        
        # 設定を保存
        import json
        with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
            
        # 訓練履歴を保存
        import pickle
        with open(save_path / 'train_history.pkl', 'wb') as f:
            pickle.dump(self.train_history, f)
            
        self.logger.info(f"モデルを保存しました: {save_path}")
        
    def load_model(self, load_path: Union[str, Path]) -> None:
        """
        モデルを読み込む.
        
        Args:
            load_path: 読み込み元のパス
        """
        load_path = Path(load_path)
        
        # モデルの状態を読み込み
        self.model.load_state_dict(torch.load(load_path / 'model.pth', map_location=self.device))
        
        # 設定を読み込み
        import json
        with open(load_path / 'config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        # 訓練履歴を読み込み
        import pickle
        history_path = load_path / 'train_history.pkl'
        if history_path.exists():
            with open(history_path, 'rb') as f:
                self.train_history = pickle.load(f)
                
        self.logger.info(f"モデルを読み込みました: {load_path}")


def create_modernbert_model(
    config: Dict,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[ModernBERTClassifier, AutoTokenizer]:
    """
    ModernBERTモデルとトークナイザーを作成する.
    
    Args:
        config: 設定辞書
        
    Returns:
        (モデル, トークナイザー)のタプル
    """
    model_name = config['model']['name']
    
    # トークナイザーを作成
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # モデルを作成
    model = ModernBERTClassifier(
        model_name=model_name,
        num_labels=config['model']['num_labels'],
        dropout_rate=float(config['model']['dropout_rate']),
        class_weights=class_weights
    )
    
    return model, tokenizer


def calculate_class_weights(labels: List[str]) -> torch.Tensor:
    """
    クラス重みを計算する.
    
    Args:
        labels: ラベルのリスト
        
    Returns:
        クラス重みのテンソル
    """
    from collections import Counter
    from sklearn.utils.class_weight import compute_class_weight
    
    # ラベルを数値に変換
    label_mapping = {'該当': 1, '非該当': 0}
    numeric_labels = [label_mapping.get(label, 0) for label in labels]
    
    # クラス重みを計算
    classes = np.unique(numeric_labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=numeric_labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float32) 