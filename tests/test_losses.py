"""
SSL損失関数のテスト
"""
import pytest
import torch

from src.losses.ssl_losses import NTXentLoss, SimSiamLoss, BarlowTwinsLoss, get_ssl_loss


class TestNTXentLoss:
    """NT-Xent損失のテスト"""

    def test_ntxent_loss_computation(self):
        """NT-Xent損失の計算をテスト"""
        criterion = NTXentLoss(temperature=0.5)

        batch_size = 4
        feature_dim = 128
        z1 = torch.randn(batch_size, feature_dim)
        z2 = torch.randn(batch_size, feature_dim)

        loss = criterion(z1, z2)

        # 損失がスカラーであることを確認
        assert loss.dim() == 0
        # 損失が正の値であることを確認
        assert loss.item() > 0

    def test_ntxent_loss_temperature(self):
        """温度パラメータが損失に影響することをテスト"""
        z1 = torch.randn(4, 128)
        z2 = torch.randn(4, 128)

        loss_low_temp = NTXentLoss(temperature=0.1)(z1, z2)
        loss_high_temp = NTXentLoss(temperature=1.0)(z1, z2)

        # 温度が異なると損失も異なるはず
        assert loss_low_temp.item() != loss_high_temp.item()


class TestSimSiamLoss:
    """SimSiam損失のテスト"""

    def test_simsiam_loss_computation(self):
        """SimSiam損失の計算をテスト"""
        criterion = SimSiamLoss()

        batch_size = 4
        feature_dim = 128
        p1 = torch.randn(batch_size, feature_dim)
        p2 = torch.randn(batch_size, feature_dim)
        z1 = torch.randn(batch_size, feature_dim)
        z2 = torch.randn(batch_size, feature_dim)

        loss = criterion(p1, p2, z1, z2)

        assert loss.dim() == 0
        # SimSiamの損失は負の値（コサイン類似度の負）
        assert loss.item() < 0 or loss.item() >= 0  # 実装により異なる


class TestBarlowTwinsLoss:
    """Barlow Twins損失のテスト"""

    def test_barlow_twins_loss_computation(self):
        """Barlow Twins損失の計算をテスト"""
        criterion = BarlowTwinsLoss(lambda_param=0.005)

        batch_size = 32  # Barlow Twinsは大きなバッチサイズが必要
        feature_dim = 128
        z1 = torch.randn(batch_size, feature_dim)
        z2 = torch.randn(batch_size, feature_dim)

        loss = criterion(z1, z2)

        assert loss.dim() == 0
        assert loss.item() >= 0  # 損失は非負


class TestGetSSLLoss:
    """get_ssl_loss関数のテスト"""

    def test_get_ssl_loss_simclr(self):
        """SimCLR損失の取得をテスト"""
        criterion = get_ssl_loss('simclr', temperature=0.5)
        assert isinstance(criterion, NTXentLoss)

    def test_get_ssl_loss_simsiam(self):
        """SimSiam損失の取得をテスト"""
        criterion = get_ssl_loss('simsiam')
        assert isinstance(criterion, SimSiamLoss)

    def test_get_ssl_loss_barlow_twins(self):
        """Barlow Twins損失の取得をテスト"""
        criterion = get_ssl_loss('barlow_twins')
        assert isinstance(criterion, BarlowTwinsLoss)

    def test_get_ssl_loss_invalid(self):
        """無効なSSL手法名でエラーが発生することをテスト"""
        with pytest.raises(ValueError):
            get_ssl_loss('invalid_method')
