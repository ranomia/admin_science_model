#!/usr/bin/env python3
"""
セットアップテストスクリプト.

このスクリプトは、プロジェクトのセットアップが正常に完了したかを確認します。
"""

import sys
import importlib
from pathlib import Path


def test_imports():
    """必要なライブラリのインポートテスト."""
    print("=== ライブラリインポートテスト ===")
    
    required_packages = [
        'pandas', 'numpy', 'torch', 'transformers',
        'sklearn', 'matplotlib', 'seaborn', 'yaml', 'tqdm', 'lightgbm'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ 以下のパッケージのインストールが必要です: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ 全ての必要パッケージが正常にインポートできました")
        return True


def test_project_structure():
    """プロジェクト構造のテスト."""
    print("\n=== プロジェクト構造テスト ===")
    
    required_paths = [
        'src/config/config.yaml',
        'src/data/preprocessing.py',
        'src/data/dataset.py',
        'src/models/baseline_models.py',
        'src/models/modernbert_classifier.py',
        'src/evaluation/metrics.py',
        'src/main.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✓ {path_str}")
        else:
            print(f"✗ {path_str}")
            missing_files.append(path_str)
    
    if missing_files:
        print(f"\n❌ 以下のファイルが見つかりません: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ 全ての必要ファイルが存在します")
        return True


def test_config_loading():
    """設定ファイルの読み込みテスト."""
    print("\n=== 設定ファイル読み込みテスト ===")
    
    try:
        import yaml
        
        config_path = Path('src/config/config.yaml')
        if not config_path.exists():
            print("❌ 設定ファイルが見つかりません")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 必要なキーの確認
        required_keys = ['data', 'model', 'training', 'evaluation', 'output']
        missing_keys = []
        
        for key in required_keys:
            if key in config:
                print(f"✓ {key}")
            else:
                print(f"✗ {key}")
                missing_keys.append(key)
        
        if missing_keys:
            print(f"\n❌ 設定ファイルに以下のキーが不足しています: {', '.join(missing_keys)}")
            return False
        else:
            print("\n✅ 設定ファイルが正常に読み込めました")
            return True
            
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みエラー: {e}")
        return False


def test_data_files():
    """データファイルの存在確認."""
    print("\n=== データファイル確認テスト ===")
    
    data_files = [
        'data/raw/train_deduplicated.csv',
        'data/raw/test_deduplicated.csv'
    ]
    
    missing_files = []
    
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            file_size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {file_path} ({file_size:.1f} MB)")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  以下のデータファイルが見つかりません: {', '.join(missing_files)}")
        print("データファイルを適切な場所に配置してください")
        return False
    else:
        print("\n✅ 全てのデータファイルが存在します")
        return True


def test_gpu_availability():
    """GPU利用可能性の確認."""
    print("\n=== GPU利用可能性テスト ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU利用可能: {gpu_count}台")
            print(f"   デバイス名: {gpu_name}")
            
            # メモリ情報
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   メモリ使用量: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB")
            
            return True
        else:
            print("⚠️  GPU利用不可 - CPUで実行されます")
            print("   ModernBERTの訓練には時間がかかる可能性があります")
            return True
            
    except Exception as e:
        print(f"❌ GPU確認エラー: {e}")
        return False


def main():
    """メイン関数."""
    print("行政事業レビュー科学技術分類モデル - セットアップテスト")
    print("=" * 60)
    
    tests = [
        ("ライブラリインポート", test_imports),
        ("プロジェクト構造", test_project_structure),
        ("設定ファイル", test_config_loading),
        ("データファイル", test_data_files),
        ("GPU利用可能性", test_gpu_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テストでエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("=== テスト結果サマリー ===")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n結果: {passed}/{total} テスト通過")
    
    if passed == total:
        print("\n🎉 全てのテストが通過しました！")
        print("以下のコマンドでプロジェクトを開始できます:")
        print("  python src/main.py --phase 1")
    else:
        print(f"\n⚠️  {total - passed}個のテストが失敗しました")
        print("上記のエラーを修正してから実行してください")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 