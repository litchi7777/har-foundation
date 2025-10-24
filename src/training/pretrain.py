"""
Self-Supervised Pre-training スクリプト

SSL手法（SimCLR、MoCo等）を用いた事前学習を実行します。
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.model import SSLModel
from src.data.dataset import PretrainDataset
from src.losses import get_ssl_loss
from src.utils.config import load_config, validate_config
from src.utils.common import set_seed, get_device, save_checkpoint, count_parameters
from src.utils.logger import setup_logger
from src.utils.training import get_optimizer, get_scheduler, init_wandb, AverageMeter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False
) -> float:
    """
    1エポック分の学習を実行

    Args:
        model: 学習するモデル
        dataloader: データローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        device: デバイス
        epoch: 現在のエポック
        use_wandb: W&Bへのログを有効化するか

    Returns:
        平均損失
    """
    model.train()
    loss_meter = AverageMeter('Loss')

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (views, _) in enumerate(pbar):
        # 2つの拡張ビュー
        view1, view2 = views[0].to(device), views[1].to(device)

        optimizer.zero_grad()

        # 順伝播
        z1, z2 = model(view1, view2)
        loss = criterion(z1, z2)

        # 逆伝播
        loss.backward()
        optimizer.step()

        # 統計を更新
        loss_meter.update(loss.item(), view1.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # W&Bにログ
        if use_wandb and batch_idx % 10 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/step': epoch * len(dataloader) + batch_idx
            })

    return loss_meter.avg


def main(args: argparse.Namespace) -> None:
    """
    メイン関数

    Args:
        args: コマンドライン引数
    """
    # 設定をロード
    config = load_config(args.config)
    validate_config(config, mode='pretrain')

    # 実験ディレクトリを作成
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = Path('experiments') / 'pretrain' / f'run_{run_id}'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint保存用ディレクトリ
    checkpoint_dir = experiment_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # ログディレクトリ
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # 設定ファイルを実験ディレクトリにコピー
    import shutil
    shutil.copy(args.config, experiment_dir / 'config.yaml')

    # ロガーをセットアップ（実験ディレクトリ内に）
    logger = setup_logger('pretrain', str(log_dir))
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Starting pre-training with config: {config['model']['name']}")

    # シード設定
    set_seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")

    # デバイス設定
    device = get_device(config['device'])
    logger.info(f"Using device: {device}")

    # データセットとデータローダーを作成
    try:
        train_dataset = PretrainDataset(
            data_path=config['data']['train_path'],
            augmentation_config=config['augmentation']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=config['data'].get('pin_memory', True),
            drop_last=True  # バッチサイズを一定に保つ
        )

        logger.info(f"Dataset size: {len(train_dataset)}")
        logger.info(f"Number of batches: {len(train_loader)}")

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    # モデルを作成
    model = SSLModel(config['model']).to(device)
    param_info = count_parameters(model)
    logger.info(
        f"Model created: {config['model']['backbone']}, "
        f"Total params: {param_info['total']:,}, "
        f"Trainable: {param_info['trainable']:,}"
    )

    # SSL損失関数を取得
    ssl_method = config.get('ssl', {}).get('method', 'simclr')
    ssl_config = config.get('ssl', {})
    criterion = get_ssl_loss(ssl_method, **ssl_config)
    logger.info(f"Using SSL method: {ssl_method}")

    # オプティマイザーを作成
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config['training'].get('optimizer', 'adam'),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )

    # スケジューラーを作成
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config['training'].get('scheduler', 'cosine'),
        T_max=config['training']['epochs']
    )

    # W&Bを初期化
    use_wandb = init_wandb(config, model)

    # トレーニングループ
    best_loss = float('inf')
    save_path = str(checkpoint_dir)

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")

        # 学習
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_wandb
        )

        logger.info(f"Epoch {epoch} - Average Loss: {train_loss:.4f}")

        # 学習率を更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")

        # W&Bにログ
        if use_wandb:
            wandb.log({
                'train/epoch_loss': train_loss,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch
            })

        # チェックポイントを保存
        save_freq = config['checkpoint'].get('save_freq', 10)
        if epoch % save_freq == 0 or epoch == config['training']['epochs']:
            is_best = train_loss < best_loss

            if is_best:
                best_loss = train_loss
                logger.info(f"New best loss: {best_loss:.4f}")

            save_file = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'loss': train_loss},
                save_path=save_path,
                is_best=is_best
            )
            logger.info(f"Checkpoint saved to {save_file}")

    # 完了
    logger.info("=" * 80)
    logger.info("Pre-training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("=" * 80)

    # クリーンアップ
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Self-Supervised Pre-training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pretrain.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()
    main(args)
