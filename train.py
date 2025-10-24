#!/usr/bin/env python
"""
HAR Foundation Training Script

Usage:
    python train.py finetune
    python train.py pretrain

Automatically detects grid search from config file.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def is_grid_search(config_path):
    """
    設定ファイルにグリッドサーチが含まれているか判定

    Returns:
        True if grid search (複数の値), False if single experiment
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    grid_search = config.get('grid_search', {})

    if not grid_search:
        return False

    # グリッドサーチパラメータをフラット化
    def has_multiple_values(d):
        """辞書内に複数の値を持つリストがあるかチェック"""
        for key, value in d.items():
            if isinstance(value, dict):
                if has_multiple_values(value):
                    return True
            elif isinstance(value, list) and len(value) > 1:
                return True
        return False

    return has_multiple_values(grid_search)


def main():
    parser = argparse.ArgumentParser(
        description='Train HAR models (auto-detects grid search)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py finetune
  python train.py pretrain

Grid search is automatically detected from config:
  - Single values in grid_search → Normal training
  - Multiple values in grid_search → Grid search
        """
    )

    parser.add_argument(
        'mode',
        choices=['finetune', 'pretrain'],
        help='Training mode: finetune or pretrain'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional, defaults to configs/{mode}.yaml)'
    )

    args = parser.parse_args()

    # Determine config path
    if args.config is None:
        config_path = f'configs/{args.mode}.yaml'
    else:
        config_path = args.config

    # Determine script path
    script_path = f'src/training/{args.mode}.py'

    print(f"\n{'='*80}")
    print(f"HAR Foundation - {args.mode.capitalize()} Training")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"{'='*80}\n")

    # Check if grid search
    if is_grid_search(config_path):
        print("🔍 Grid search detected - running multiple experiments\n")

        from src.training.run_experiments import main as run_experiments

        class ExperimentArgs:
            def __init__(self, config, script):
                self.config = config
                self.script = script

        exp_args = ExperimentArgs(config_path, script_path)
        run_experiments(exp_args)
    else:
        print("▶ Single experiment - running normal training\n")

        # Import and run the training script directly
        if args.mode == 'pretrain':
            from src.training.pretrain import main as train_main
        else:  # finetune
            from src.training.finetune import main as train_main

        class TrainArgs:
            def __init__(self, config):
                self.config = config

        train_args = TrainArgs(config_path)
        train_main(train_args)


if __name__ == '__main__':
    main()
