"""
モデルのテスト
"""
import pytest
import torch

from src.models.model import SSLModel, ClassificationModel
from src.models.backbones import get_backbone, freeze_backbone


class TestBackbones:
    """バックボーン関連のテスト"""

    def test_get_backbone_resnet18(self):
        """ResNet18バックボーンの取得をテスト"""
        backbone, feature_dim = get_backbone('resnet18')
        assert backbone is not None
        assert feature_dim == 512

    def test_get_backbone_resnet50(self):
        """ResNet50バックボーンの取得をテスト"""
        backbone, feature_dim = get_backbone('resnet50')
        assert backbone is not None
        assert feature_dim == 2048

    def test_get_backbone_invalid(self):
        """無効なバックボーン名でエラーが発生することをテスト"""
        with pytest.raises(ValueError):
            get_backbone('invalid_backbone')

    def test_freeze_backbone(self):
        """バックボーン凍結をテスト"""
        backbone, _ = get_backbone('resnet18')
        freeze_backbone(backbone)

        # すべてのパラメータが凍結されているか確認
        for param in backbone.parameters():
            assert not param.requires_grad


class TestSSLModel:
    """SSLモデルのテスト"""

    def test_ssl_model_creation(self):
        """SSLモデルの作成をテスト"""
        config = {
            'backbone': 'resnet18',
            'feature_dim': 128,
            'projection_dim': 64
        }
        model = SSLModel(config)
        assert model is not None

    def test_ssl_model_forward(self):
        """SSLモデルの順伝播をテスト"""
        config = {
            'backbone': 'resnet18',
            'feature_dim': 128,
            'projection_dim': 64
        }
        model = SSLModel(config)

        # ダミー入力
        batch_size = 4
        x1 = torch.randn(batch_size, 3, 224, 224)
        x2 = torch.randn(batch_size, 3, 224, 224)

        # 順伝播
        z1, z2 = model(x1, x2)

        # 出力形状を確認
        assert z1.shape == (batch_size, 64)
        assert z2.shape == (batch_size, 64)


class TestClassificationModel:
    """分類モデルのテスト"""

    def test_classification_model_creation(self):
        """分類モデルの作成をテスト"""
        config = {
            'backbone': 'resnet18',
            'num_classes': 10
        }
        model = ClassificationModel(config)
        assert model is not None

    def test_classification_model_forward(self):
        """分類モデルの順伝播をテスト"""
        config = {
            'backbone': 'resnet18',
            'num_classes': 10
        }
        model = ClassificationModel(config)

        # ダミー入力
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        # 順伝播
        logits = model(x)

        # 出力形状を確認
        assert logits.shape == (batch_size, 10)

    def test_classification_model_with_dropout(self):
        """ドロップアウト付き分類モデルをテスト"""
        config = {
            'backbone': 'resnet18',
            'num_classes': 10,
            'dropout': 0.5
        }
        model = ClassificationModel(config)
        assert model.dropout is not None
