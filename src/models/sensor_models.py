"""
時系列センサーデータ用のモデル

1D CNNベースのHARモデルを提供します。
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """1D Convolutional Block with BatchNorm and ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM: 1D CNN + LSTM for time-series classification

    Reference: Ordóñez and Roggen (2016)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: Tuple[int, ...] = (64, 128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        # Convolutional layers
        conv_layers = []
        prev_channels = in_channels
        for channels in conv_channels:
            conv_layers.append(Conv1DBlock(prev_channels, channels, kernel_size=5, padding=2))
            prev_channels = channels

        self.conv = nn.Sequential(*conv_layers)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Conv layers
        x = self.conv(x)  # (batch, channels, time_steps)

        # Reshape for LSTM: (batch, time_steps, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        x, _ = self.lstm(x)  # (batch, time_steps, hidden)

        # Take last timestep
        x = x[:, -1, :]  # (batch, hidden)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class SimpleCNN(nn.Module):
    """
    Simple 1D CNN for time-series classification

    軽量で高速なベースラインモデル
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()

        # Convolutional blocks
        blocks = []
        prev_channels = in_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            blocks.append(Conv1DBlock(prev_channels, out_channels, kernel_size=5, padding=2, dropout=dropout))
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_channels = out_channels

        self.conv_blocks = nn.Sequential(*blocks)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc = nn.Linear(prev_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.conv_blocks(x)  # (batch, channels, reduced_time)
        x = self.gap(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        x = self.fc(x)  # (batch, num_classes)
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual Block"""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = Conv1DBlock(channels, channels, kernel_size=3, padding=1, dropout=dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for time-series classification

    より深いモデルで複雑なパターンを学習
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        num_res_blocks: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        # Initial convolution
        self.init_conv = Conv1DBlock(in_channels, base_channels, kernel_size=7, padding=3)

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock1D(base_channels, dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Downsample
        self.downsample = nn.Sequential(
            Conv1DBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, dropout=dropout),
            Conv1DBlock(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, dropout=dropout),
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.downsample(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class SensorSSLModel(nn.Module):
    """
    Self-Supervised Learning用のセンサーモデル

    エンコーダー + プロジェクションヘッド
    """

    def __init__(
        self,
        in_channels: int,
        backbone: str = "simple_cnn",
        projection_dim: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Encoder (backbone)
        if backbone == "simple_cnn":
            self.encoder = SimpleCNN(in_channels, num_classes=hidden_dim, dropout=0.0)
            # Remove final FC layer
            self.encoder.fc = nn.Identity()
            encoder_dim = self.encoder.conv_blocks[-2].conv.out_channels  # Get last conv channels
        elif backbone == "resnet1d":
            self.encoder = ResNet1D(in_channels, num_classes=hidden_dim, dropout=0.0)
            self.encoder.fc = nn.Identity()
            encoder_dim = self.encoder.downsample[-1].conv.out_channels
        elif backbone == "deepconvlstm":
            self.encoder = DeepConvLSTM(in_channels, num_classes=hidden_dim, dropout=0.0)
            self.encoder.fc = nn.Identity()
            encoder_dim = self.encoder.lstm.hidden_size
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x1: First view (batch_size, channels, time_steps)
            x2: Second view (optional)

        Returns:
            (z1, z2): Projected representations
        """
        # Encode
        h1 = self.encoder(x1)

        # Global pooling if needed (encoder might return feature maps)
        if len(h1.shape) > 2:
            h1 = torch.mean(h1, dim=-1)

        # Project
        z1 = self.projection(h1)

        if x2 is not None:
            h2 = self.encoder(x2)
            if len(h2.shape) > 2:
                h2 = torch.mean(h2, dim=-1)
            z2 = self.projection(h2)
            return z1, z2

        return z1, None


class SensorClassificationModel(nn.Module):
    """
    Fine-tuning用のセンサー分類モデル

    事前学習済みエンコーダー + 分類ヘッド
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone: str = "simple_cnn",
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Encoder (backbone)
        if backbone == "simple_cnn":
            self.encoder = SimpleCNN(in_channels, num_classes, dropout=0.5)
            encoder_dim = self.encoder.conv_blocks[-2].conv.out_channels
            self.encoder.fc = nn.Identity()  # Remove classification head
        elif backbone == "resnet1d":
            self.encoder = ResNet1D(in_channels, num_classes, dropout=0.5)
            encoder_dim = self.encoder.downsample[-1].conv.out_channels
            self.encoder.fc = nn.Identity()
        elif backbone == "deepconvlstm":
            self.encoder = DeepConvLSTM(in_channels, num_classes, dropout=0.5)
            encoder_dim = self.encoder.lstm.hidden_size
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.fc = nn.Linear(encoder_dim, num_classes)

    def _load_pretrained(self, path: str):
        """事前学習済みの重みをロード"""
        checkpoint = torch.load(path, map_location='cpu')

        # SSL model の encoder の重みを抽出
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # encoder.* の重みのみを抽出
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state[new_key] = value

        # Load into encoder
        self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded pretrained weights from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode
        h = self.encoder(x)

        # Global pooling if needed
        if len(h.shape) > 2:
            h = torch.mean(h, dim=-1)

        # Classify
        logits = self.fc(h)

        return logits


def get_sensor_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    センサーモデルのファクトリー関数

    Args:
        model_name: "simple_cnn", "resnet1d", "deepconvlstm"
        in_channels: 入力チャンネル数
        num_classes: クラス数
        **kwargs: モデル固有のパラメータ

    Returns:
        モデルインスタンス
    """
    if model_name == "simple_cnn":
        return SimpleCNN(in_channels, num_classes, **kwargs)
    elif model_name == "resnet1d":
        return ResNet1D(in_channels, num_classes, **kwargs)
    elif model_name == "deepconvlstm":
        return DeepConvLSTM(in_channels, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
