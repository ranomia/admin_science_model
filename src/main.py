"""
行政事業レビューデータを用いた科学技術事業分類モデル構築のメインスクリプト.

Phase 1: データ探索・前処理・ベースライン構築
Phase 2: ModernBERTファインチューニング・ハイパーパラメータ調整
Phase 3: モデル評価・比較分析・結果解釈
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# 自作モジュールのインポート
from data.preprocessing import DataPreprocessor, setup_logging
from models.baseline_models import create_baseline_models, BaselineEvaluator
from models.modernbert_classifier import create_modernbert_model, ModernBERTTrainer
from models.ensemble_model import EnsembleModel
from data.dataset import create_data_loaders
from evaluation.metrics import ModelEvaluator, MetricsCalculator

# 警告を抑制
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> Dict:
    """
    設定ファイルを読み込む.
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        設定辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def phase1_data_exploration_and_baseline(config: Dict, logger: logging.Logger) -> Dict:
    """
    Phase 1: データ探索・前処理・ベースライン構築.
    
    Args:
        config: 設定辞書
        logger: ロガー
        
    Returns:
        Phase 1の結果辞書
    """
    logger.info("=" * 50)
    logger.info("Phase 1: データ探索・前処理・ベースライン構築を開始します")
    logger.info("=" * 50)
    
    # データ前処理器を初期化
    preprocessor = DataPreprocessor(config)
    
    # 1. データ読み込み
    logger.info("1. データ読み込みを開始します")
    train_df = preprocessor.load_data(config['data']['train_path'])
    test_df = preprocessor.load_data(config['data']['test_path'])
    
    # 2. データ分析（訓練データのみ）
    logger.info("2. データ分析を開始します（訓練データのみ）")
    train_analysis = preprocessor.analyze_data(train_df)
    
    # 分析結果をログ出力
    logger.info(f"訓練データ形状: {train_analysis['shape']}")
    logger.info(f"テストデータ形状: {test_df.shape}")  # 形状のみ表示
    logger.info(f"訓練データの目的変数分布: {train_analysis.get('target_distribution', 'N/A')}")
    
    # 欠損値情報
    logger.info("訓練データの欠損率:")
    for col, rate in train_analysis['missing_rates'].items():
        if rate > 0:
            logger.info(f"  {col}: {rate:.2%}")
    
    # 3. データ前処理
    logger.info("3. データ前処理を開始します")
    train_processed = preprocessor.preprocess_pipeline(train_df)
    test_processed = preprocessor.preprocess_pipeline(test_df)
    
    # 4. データ分割（訓練・検証）
    logger.info("4. データ分割を開始します")
    train_split, val_split = preprocessor.split_data(train_processed)
    
    # 5. 前処理済みデータを保存
    logger.info("5. 前処理済みデータを保存します")
    preprocessor.save_processed_data(train_split, 'train_processed.csv')
    preprocessor.save_processed_data(val_split, 'val_processed.csv')
    preprocessor.save_processed_data(test_processed, 'test_processed.csv')
    
    # 6. ベースラインモデル構築
    logger.info("6. ベースラインモデル構築を開始します")
    baseline_models = create_baseline_models(config)
    baseline_evaluator = BaselineEvaluator(config)
    
    # 特徴量とラベルを準備
    X_train = train_split['combined_text']
    y_train = train_split[config['data']['target_column']]
    X_val = val_split['combined_text']
    y_val = val_split[config['data']['target_column']]
    X_test = test_processed['combined_text']
    y_test = test_processed[config['data']['target_column']] if config['data']['target_column'] in test_processed.columns else None
    
    # 7. ベースラインモデル比較
    logger.info("7. ベースラインモデル比較を開始します")
    baseline_results = baseline_evaluator.compare_models(
        baseline_models, X_train, y_train, X_val, y_val
    )
    
    logger.info("ベースラインモデル比較結果:")
    for _, row in baseline_results.iterrows():
        logger.info(f"  {row['model']} - {row['metric']}: {row['val_score']:.4f}")
    
    # 8. 最良のベースラインモデルで詳細評価（検証データのみ）
    logger.info("8. 最良のベースラインモデルで詳細評価を実行します（検証データ）")
    best_baseline_name = baseline_results.groupby('model')['val_score'].mean().idxmax()
    best_baseline_model = baseline_models[best_baseline_name]
    
    # 最良モデルを再訓練
    best_baseline_model.fit(X_train, y_train)
    
    # 詳細レポート生成（検証データで評価）
    detailed_report = baseline_evaluator.generate_detailed_report(
        best_baseline_model, X_val, y_val, best_baseline_name
    )
    
    logger.info(f"最良ベースラインモデル ({best_baseline_name}) の検証データ結果:")
    for metric, score in detailed_report['metrics'].items():
        logger.info(f"  {metric}: {score:.4f}")
    
    # Phase 1結果をまとめる
    phase1_results = {
        'train_analysis': train_analysis,
        'baseline_comparison': baseline_results.to_dict(),
        'best_baseline_model': best_baseline_name,
        'best_baseline_metrics': detailed_report['metrics'],
        'data_shapes': {
            'train': list(train_split.shape),
            'val': list(val_split.shape),
            'test': list(test_processed.shape)
        }
    }
    
    logger.info("Phase 1が完了しました")
    return phase1_results


def phase2_modernbert_training(config: Dict, logger: logging.Logger) -> Dict:
    """
    Phase 2: ModernBERTファインチューニング・ハイパーパラメータ調整.
    
    Args:
        config: 設定辞書
        logger: ロガー
        
    Returns:
        Phase 2の結果辞書
    """
    logger.info("=" * 50)
    logger.info("Phase 2: ModernBERTファインチューニングを開始します")
    logger.info("=" * 50)
    
    # GPU利用可能性を確認
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    # 前処理済みデータを読み込み
    processed_dir = Path(config['data']['processed_dir'])
    train_df = pd.read_csv(processed_dir / 'train_processed.csv')
    val_df = pd.read_csv(processed_dir / 'val_processed.csv')
    
    # ModernBERTモデルとトークナイザーを作成
    logger.info("ModernBERTモデルを初期化します")
    model, tokenizer = create_modernbert_model(config)
    
    # データローダーを作成
    logger.info("データローダーを作成します")
    data_loaders = create_data_loaders(
        train_df, val_df, None, tokenizer, config  # テストデータは除外
    )
    
    # トレーナーを初期化
    trainer = ModernBERTTrainer(model, tokenizer, config, device)
    
    # モデル保存ディレクトリを作成
    model_dir = Path(config['output']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練実行（訓練・検証分割で）
    logger.info("ModernBERTの訓練を開始します（訓練・検証分割で）")
    train_history = trainer.train(
        data_loaders['train'],
        data_loaders.get('val'),
        save_dir=model_dir
    )
    
    # 最終的に訓練+検証データで再訓練（テストデータ評価用）
    logger.info("最終訓練: 訓練+検証データを結合して再訓練します")
    
    # 訓練+検証データを結合
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # 最終訓練用データローダーを作成
    final_data_loaders = create_data_loaders(
        train_val_df, None, None, tokenizer, config
    )
    
    # 最良モデルを再度ロード
    trainer.load_model(model_dir / 'best_model')
    
    # 最終訓練実行（Early Stoppingなし、少ないエポック数で）
    final_config = config.copy()
    final_config['training']['num_epochs'] = 1  # 最終調整のみ
    final_trainer = ModernBERTTrainer(trainer.model, tokenizer, final_config, device)
    
    final_train_history = final_trainer.train(
        final_data_loaders['train'],
        None,  # 検証データなし
        save_dir=model_dir / 'final_model_dir'
    )
    
    # 最終モデルを保存
    final_trainer.save_model(model_dir / 'final_model')
    
    # Phase 2ではテストデータでの評価は行わない
    logger.info("Phase 2: 訓練が完了しました。テストデータでの評価はPhase 3で実行します。")
    
    phase2_results = {
        'train_history': train_history,
        'model_path': str(model_dir / 'best_model'),
        'device': str(device)
    }
    
    logger.info("Phase 2が完了しました")
    return phase2_results


def phase3_model_evaluation(config: Dict, logger: logging.Logger, phase1_results: Dict, phase2_results: Dict) -> Dict:
    """
    Phase 3: モデル評価・比較分析・結果解釈.
    
    Args:
        config: 設定辞書
        logger: ロガー
        phase1_results: Phase 1の結果
        phase2_results: Phase 2の結果
        
    Returns:
        Phase 3の結果辞書
    """
    logger.info("=" * 50)
    logger.info("Phase 3: モデル評価・比較分析を開始します")
    logger.info("=" * 50)
    
    # 結果ディレクトリを作成
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # モデル評価器を初期化
    evaluator = ModelEvaluator(results_dir)
    
    # 前処理済みデータを読み込み
    processed_dir = Path(config['data']['processed_dir'])
    test_df = pd.read_csv(processed_dir / 'test_processed.csv')
    
    # テストデータの準備
    X_test = test_df['combined_text']
    y_test = test_df[config['data']['target_column']]
    
    logger.info("1. ベースラインモデルをテストデータで評価します")
    # ベースラインモデルを再作成して訓練
    baseline_models = create_baseline_models(config)
    best_baseline_name = phase1_results['best_baseline_model']
    baseline_model = baseline_models[best_baseline_name]
    
    # 前処理済み訓練・検証データを読み込み（テストデータには触れない）
    train_df = pd.read_csv(processed_dir / 'train_processed.csv')
    val_df = pd.read_csv(processed_dir / 'val_processed.csv')
    
    # 訓練データと検証データを結合してベースラインモデルを最終訓練
    X_train = train_df['combined_text']
    y_train = train_df[config['data']['target_column']]
    X_val = val_df['combined_text']
    y_val = val_df[config['data']['target_column']]
    
    # 訓練データと検証データを結合
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    # ベースラインモデルを訓練+検証データで最終訓練
    logger.info(f"ベースラインモデル ({best_baseline_name}) を訓練+検証データで最終訓練します")
    baseline_model.fit(X_train_val, y_train_val)

    # テストデータで評価
    baseline_evaluator = BaselineEvaluator(config)
    baseline_test_report = baseline_evaluator.generate_detailed_report(
        baseline_model, X_test, y_test, best_baseline_name
    )

    logger.info(f"ベースラインモデル ({phase1_results['best_baseline_model']}) のテストデータ結果:")
    for metric, score in baseline_test_report['metrics'].items():
        logger.info(f"  {metric}: {score:.4f}")

    logger.info("2. ModernBERTモデルをテストデータで評価します")
    # ModernBERTモデルをテストデータで評価
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 最良のModernBERTモデルをロード
    model, tokenizer = create_modernbert_model(config)
    trainer = ModernBERTTrainer(model, tokenizer, config, device)

    # ベストモデルをロード
    model_path = Path(phase2_results['model_path'])
    trainer.load_model(model_path)

    # テストデータローダーを作成
    test_data_loaders = create_data_loaders(
        None, None, test_df, tokenizer, config
    )

    # テストデータで評価
    modernbert_test_metrics = trainer.evaluate(test_data_loaders['test'])

    logger.info("ModernBERTモデルのテストデータ評価結果:")
    for metric, score in modernbert_test_metrics.items():
        logger.info(f"  {metric}: {score:.4f}")

    # アンサンブルモデルの評価
    logger.info("3. ベースラインとModernBERTのアンサンブルを評価します")
    ensemble_model = EnsembleModel(baseline_model, trainer)
    ensemble_preds, ensemble_probs = ensemble_model.predict(
        X_test, test_data_loaders['test']
    )
    metrics_calc = MetricsCalculator()
    y_true_encoded = baseline_model._encode_labels(y_test)
    ensemble_metrics = metrics_calc.calculate_all_metrics(
        y_true_encoded, ensemble_preds, ensemble_probs
    )
    logger.info("アンサンブルモデルのテストデータ評価結果:")
    for metric, score in ensemble_metrics.items():
        logger.info(f"  {metric}: {score:.4f}")

    # ベースライン・ModernBERT・アンサンブルの比較
    logger.info("4. 最終的なモデル比較を実行します")

    # 結果をまとめる
    comparison_results = {
        'baseline_best_model': phase1_results['best_baseline_model'],
        'baseline_test_metrics': baseline_test_report['metrics'],
        'modernbert_test_metrics': modernbert_test_metrics,
        'ensemble_test_metrics': ensemble_metrics,
        'improvement': {'modernbert': {}, 'ensemble': {}},
    }

    # 改善度を計算
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in baseline_test_report['metrics']:
            baseline_score = baseline_test_report['metrics'][metric]
            if metric in modernbert_test_metrics:
                modernbert_score = modernbert_test_metrics[metric]
                comparison_results['improvement']['modernbert'][metric] = {
                    'absolute': modernbert_score - baseline_score,
                    'relative': (modernbert_score - baseline_score) / baseline_score * 100
                    if baseline_score > 0
                    else 0,
                }
            if metric in ensemble_metrics:
                ensemble_score = ensemble_metrics[metric]
                comparison_results['improvement']['ensemble'][metric] = {
                    'absolute': ensemble_score - baseline_score,
                    'relative': (ensemble_score - baseline_score) / baseline_score * 100
                    if baseline_score > 0
                    else 0,
                }

    logger.info("最終モデル比較結果（テストデータ）:")
    logger.info(f"ベースライン最良モデル: {comparison_results['baseline_best_model']}")
    logger.info("性能比較:")
    for model_name, metrics_dict in comparison_results['improvement'].items():
        logger.info(f"{model_name}:")
        for metric, imp in metrics_dict.items():
            logger.info(f"  {metric}: {imp['absolute']:+.4f} ({imp['relative']:+.2f}%)")

    phase3_results = {
        'model_comparison': comparison_results,
        'results_saved_to': str(results_dir)
    }
    
    logger.info("Phase 3が完了しました")
    return phase3_results


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description='行政事業レビュー科学技術分類モデル構築')
    parser.add_argument('--config', default='src/config/config.yaml', help='設定ファイルのパス')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all', help='実行するフェーズ')
    parser.add_argument('--log-level', default='INFO', help='ログレベル')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 設定読み込み
    config = load_config(args.config)
    logger.info(f"設定ファイルを読み込みました: {args.config}")
    
    # 出力ディレクトリを作成
    for output_dir in ['model_dir', 'results_dir', 'logs_dir', 'figures_dir']:
        Path(config['output'][output_dir]).mkdir(parents=True, exist_ok=True)
    
    # 結果を保存する辞書
    all_results = {}
    
    try:
        if args.phase in ['1', 'all']:
            phase1_results = phase1_data_exploration_and_baseline(config, logger)
            all_results['phase1'] = phase1_results
            
            # Phase 1結果を保存
            with open(Path(config['output']['results_dir']) / 'phase1_results.json', 'w', encoding='utf-8') as f:
                json.dump(phase1_results, f, ensure_ascii=False, indent=2, default=str)
        
        if args.phase in ['2', 'all']:
            phase2_results = phase2_modernbert_training(config, logger)
            all_results['phase2'] = phase2_results
            
            # Phase 2結果を保存
            with open(Path(config['output']['results_dir']) / 'phase2_results.json', 'w', encoding='utf-8') as f:
                json.dump(phase2_results, f, ensure_ascii=False, indent=2, default=str)
        
        if args.phase in ['3', 'all']:
            # Phase 1, 2の結果が必要
            if 'phase1' not in all_results:
                with open(Path(config['output']['results_dir']) / 'phase1_results.json', 'r', encoding='utf-8') as f:
                    all_results['phase1'] = json.load(f)
            
            if 'phase2' not in all_results:
                with open(Path(config['output']['results_dir']) / 'phase2_results.json', 'r', encoding='utf-8') as f:
                    all_results['phase2'] = json.load(f)
            
            phase3_results = phase3_model_evaluation(
                config, logger, all_results['phase1'], all_results['phase2']
            )
            all_results['phase3'] = phase3_results
            
            # Phase 3結果を保存
            with open(Path(config['output']['results_dir']) / 'phase3_results.json', 'w', encoding='utf-8') as f:
                json.dump(phase3_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 全結果を保存
        with open(Path(config['output']['results_dir']) / 'all_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("=" * 50)
        logger.info("全ての処理が正常に完了しました!")
        logger.info(f"結果は {config['output']['results_dir']} に保存されました")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise


if __name__ == '__main__':
    main() 
