"""
ユーティリティ関数のテスト
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.utils.common import set_seed, count_parameters, format_time
from src.utils.config import load_config, ConfigValidationError
from src.utils.training import get_optimizer, get_scheduler, AverageMeter, EarlyStopping


class TestCommonUtils:
    """共通ユーティリティのテスト"""

    def test_set_seed(self):
        """シード設定のテスト"""
        # シードを設定して乱数生成
        set_seed(42)
        val1 = torch.rand(1).item()

        # 同じシードで再度生成
        set_seed(42)
        val2 = torch.rand(1).item()

        # 同じ値になることを確認
        assert val1 == val2

    def test_count_parameters(self):
        """パラメータカウントのテスト"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        param_info = count_parameters(model)

        assert 'total' in param_info
        assert 'trainable' in param_info
        assert 'non_trainable' in param_info
        assert param_info['total'] > 0
        assert param_info['trainable'] == param_info['total']

    def test_format_time(self):
        """時間フォーマットのテスト"""
        assert format_time(45) == "45s"
        assert format_time(125) == "2m 5s"
        assert format_time(3665) == "1h 1m 5s"


class TestConfigUtils:
    """設定ユーティリティのテスト"""

    def test_load_config_nonexistent(self):
        """存在しない設定ファイルでエラーが発生することをテスト"""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent.yaml')


class TestTrainingUtils:
    """トレーニングユーティリティのテスト"""

    def test_get_optimizer_adam(self):
        """Adamオプティマイザーの取得をテスト"""
        model = nn.Linear(10, 5)
        optimizer = get_optimizer(model, 'adam', 0.001)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_sgd(self):
        """SGDオプティマイザーの取得をテスト"""
        model = nn.Linear(10, 5)
        optimizer = get_optimizer(model, 'sgd', 0.001)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_get_optimizer_adamw(self):
        """AdamWオプティマイザーの取得をテスト"""
        model = nn.Linear(10, 5)
        optimizer = get_optimizer(model, 'adamw', 0.001)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_get_optimizer_invalid(self):
        """無効なオプティマイザー名でエラーが発生することをテスト"""
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError):
            get_optimizer(model, 'invalid_optimizer', 0.001)

    def test_get_scheduler_step(self):
        """StepLRスケジューラーの取得をテスト"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = get_scheduler(optimizer, 'step', step_size=10)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_get_scheduler_cosine(self):
        """CosineAnnealingLRスケジューラーの取得をテスト"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = get_scheduler(optimizer, 'cosine', T_max=100)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


class TestAverageMeter:
    """AverageMeterのテスト"""

    def test_average_meter_basic(self):
        """基本的な平均計算をテスト"""
        meter = AverageMeter('test')

        meter.update(10, 1)
        assert meter.avg == 10

        meter.update(20, 1)
        assert meter.avg == 15

    def test_average_meter_reset(self):
        """リセット機能をテスト"""
        meter = AverageMeter('test')
        meter.update(10, 1)
        meter.reset()

        assert meter.avg == 0
        assert meter.count == 0


class TestEarlyStopping:
    """EarlyStoppingのテスト"""

    def test_early_stopping_min_mode(self):
        """最小化モードのearly stoppingをテスト"""
        early_stop = EarlyStopping(patience=2, mode='min')

        # 損失が改善している場合
        assert not early_stop(1.0)  # best_score = 1.0, counter = 0
        assert not early_stop(0.9)  # improved, best_score = 0.9, counter = 0
        assert not early_stop(0.8)  # improved, best_score = 0.8, counter = 0

        # 損失が改善しない場合
        assert not early_stop(0.85)  # not improved, counter = 1
        assert early_stop(0.86)  # not improved, counter = 2 >= patience

    def test_early_stopping_max_mode(self):
        """最大化モードのearly stoppingをテスト"""
        early_stop = EarlyStopping(patience=2, mode='max')

        # 精度が改善している場合
        assert not early_stop(0.8)  # best_score = 0.8, counter = 0
        assert not early_stop(0.9)  # improved, best_score = 0.9, counter = 0
        assert not early_stop(0.95)  # improved, best_score = 0.95, counter = 0

        # 精度が改善しない場合
        assert not early_stop(0.94)  # not improved, counter = 1
        assert early_stop(0.93)  # not improved, counter = 2 >= patience
