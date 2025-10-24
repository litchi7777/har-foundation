"""
トレーニングの統合テスト

実際のトレーニングフローが正しく動作するかを簡単なデータで確認します。
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path

from src.models.sensor_models import SimpleCNN, Resnet, SensorClassificationModel
from src.utils.training import get_optimizer, get_scheduler, AverageMeter


class TestTrainingIntegration:
    """トレーニングの統合テスト"""

    @pytest.fixture
    def dummy_data(self):
        """ダミーデータを作成"""
        # 小規模なデータセット (100サンプル)
        batch_size = 16
        n_samples = 100
        in_channels = 9
        seq_length = 150
        num_classes = 5

        # ランダムデータ
        X = torch.randn(n_samples, in_channels, seq_length)
        y = torch.randint(0, num_classes, (n_samples,))

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return {
            'dataloader': dataloader,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'seq_length': seq_length
        }

    def test_simple_cnn_training_loop(self, dummy_data):
        """SimpleCNNでトレーニングループが動作するか確認"""
        model = SimpleCNN(
            in_channels=dummy_data['in_channels'],
            num_classes=dummy_data['num_classes']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 1エポック分トレーニング
        model.train()
        loss_meter = AverageMeter('Loss')

        for batch_idx, (data, target) in enumerate(dummy_data['dataloader']):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), data.size(0))

        # 損失が計算されていることを確認
        assert loss_meter.count > 0
        assert loss_meter.avg > 0
        print(f"SimpleCNN - Average loss: {loss_meter.avg:.4f}")

    def test_resnet_training_loop(self, dummy_data):
        """カスタムResNetでトレーニングループが動作するか確認"""
        model = SensorClassificationModel(
            in_channels=dummy_data['in_channels'],
            num_classes=dummy_data['num_classes'],
            backbone='resnet',
            foundationUK=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 1エポック分トレーニング
        model.train()
        loss_meter = AverageMeter('Loss')

        for batch_idx, (data, target) in enumerate(dummy_data['dataloader']):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), data.size(0))

        # 損失が計算されていることを確認
        assert loss_meter.count > 0
        assert loss_meter.avg > 0
        print(f"Resnet - Average loss: {loss_meter.avg:.4f}")

    def test_optimizer_and_scheduler(self, dummy_data):
        """オプティマイザーとスケジューラーが正しく動作するか確認"""
        model = SimpleCNN(
            in_channels=dummy_data['in_channels'],
            num_classes=dummy_data['num_classes']
        )

        # オプティマイザーとスケジューラーを作成
        optimizer = get_optimizer(
            model=model,
            optimizer_name='adam',
            learning_rate=0.001,
            weight_decay=0.0001
        )

        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name='step',
            step_size=5,
            gamma=0.5
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # 数エポック分トレーニング
        for epoch in range(10):
            # 簡単なトレーニングステップ
            for data, target in dummy_data['dataloader']:
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

            # スケジューラーをステップ
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            # 5エポック目で学習率が下がることを確認
            if epoch == 5:
                assert current_lr < initial_lr
                print(f"Learning rate decreased from {initial_lr} to {current_lr}")

    def test_overfitting_single_batch(self, dummy_data):
        """単一バッチでオーバーフィットできるか確認（モデルが学習可能かの基本チェック）"""
        model = SimpleCNN(
            in_channels=dummy_data['in_channels'],
            num_classes=dummy_data['num_classes']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 単一バッチを取得
        data, target = next(iter(dummy_data['dataloader']))

        # 複数回トレーニング
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # 損失が減少していることを確認
        assert losses[-1] < losses[0], "Model should overfit to single batch"
        print(f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    def test_model_evaluation_mode(self, dummy_data):
        """評価モードが正しく動作するか確認"""
        model = SimpleCNN(
            in_channels=dummy_data['in_channels'],
            num_classes=dummy_data['num_classes'],
            dropout=0.5  # ドロップアウト有り
        )

        # 評価モード
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in dummy_data['dataloader']:
                output = model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())

        # 予測が生成されていることを確認
        assert len(all_preds) > 0
        assert len(all_preds) == len(all_targets)
        print(f"Evaluated {len(all_preds)} samples")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
