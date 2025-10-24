"""
Self-Supervised Learning (SSL) 損失関数

SimCLR、MoCo、BYOL等のSSL手法で使用される損失関数を実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent)

    SimCLRで使用される対照学習損失関数。
    同じサンプルの異なる拡張ビュー間の類似度を最大化し、
    異なるサンプル間の類似度を最小化する。

    References:
        Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations"
        https://arxiv.org/abs/2002.05709
    """

    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            temperature: スケーリング温度パラメータ（小さいほど hard negative mining）
            reduction: 損失の集約方法 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent損失を計算

        Args:
            z1: 第1ビューの特徴ベクトル [batch_size, feature_dim]
            z2: 第2ビューの特徴ベクトル [batch_size, feature_dim]

        Returns:
            損失値
        """
        batch_size = z1.size(0)
        device = z1.device

        # 特徴ベクトルを正規化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 全ペアの表現を結合 [2*batch_size, feature_dim]
        z = torch.cat([z1, z2], dim=0)

        # 類似度行列を計算 [2*batch_size, 2*batch_size]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # 対角成分をマスク（自分自身との類似度を除外）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # 正例ペアのインデックスを作成
        # (i, i+batch_size) と (i+batch_size, i) が正例ペア
        positive_indices = torch.arange(batch_size, device=device)
        positive_indices = torch.cat([
            positive_indices + batch_size,  # z1のpositive
            positive_indices  # z2のpositive
        ])

        # 正例の類似度を取得
        pos_sim = torch.cat([
            torch.diag(sim_matrix[:batch_size, batch_size:]),
            torch.diag(sim_matrix[batch_size:, :batch_size])
        ])

        # LogSumExp計算による安定化
        # loss = -log(exp(pos) / sum(exp(all)))
        #      = -pos + log(sum(exp(all)))
        numerator = pos_sim
        denominator = torch.logsumexp(sim_matrix, dim=1)

        loss = -numerator + denominator

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SimSiamLoss(nn.Module):
    """
    SimSiam損失関数

    Negative pairsなしの対照学習。
    予測器を用いて一方の表現からもう一方を予測する。

    References:
        Chen & He "Exploring Simple Siamese Representation Learning"
        https://arxiv.org/abs/2011.10566
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        SimSiam損失を計算

        Args:
            p1: 第1ビューの予測 [batch_size, feature_dim]
            p2: 第2ビューの予測 [batch_size, feature_dim]
            z1: 第1ビューの特徴（stop gradient） [batch_size, feature_dim]
            z2: 第2ビューの特徴（stop gradient） [batch_size, feature_dim]

        Returns:
            損失値
        """
        # コサイン類似度を使用した対称的な損失
        loss = -(
            self._cosine_similarity(p1, z2.detach()).mean() +
            self._cosine_similarity(p2, z1.detach()).mean()
        ) * 0.5

        return loss

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """コサイン類似度を計算"""
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x * y).sum(dim=-1)


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins損失関数

    2つのビューの埋め込み間の相関行列を単位行列に近づける。
    対角成分は1（同じ特徴は相関）、非対角成分は0（異なる特徴は無相関）。

    References:
        Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
        https://arxiv.org/abs/2103.03230
    """

    def __init__(self, lambda_param: float = 0.005):
        """
        Args:
            lambda_param: 非対角要素の重み（redundancy reduction項の重み）
        """
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Barlow Twins損失を計算

        Args:
            z1: 第1ビューの特徴 [batch_size, feature_dim]
            z2: 第2ビューの特徴 [batch_size, feature_dim]

        Returns:
            損失値
        """
        batch_size = z1.size(0)

        # 特徴を正規化（バッチ単位で平均0、分散1）
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-8)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-8)

        # 相関行列を計算 [feature_dim, feature_dim]
        cross_corr = torch.mm(z1_norm.t(), z2_norm) / batch_size

        # 損失 = 対角成分を1に近づける + 非対角成分を0に近づける
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cross_corr).pow_(2).sum()

        loss = on_diag + self.lambda_param * off_diag

        return loss

    @staticmethod
    def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        """非対角要素を取得"""
        n = matrix.size(0)
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_ssl_loss(method: str, **kwargs) -> nn.Module:
    """
    SSL手法名から損失関数を取得

    Args:
        method: SSL手法名 ('simclr', 'simsiam', 'barlow_twins')
        **kwargs: 損失関数固有のパラメータ

    Returns:
        損失関数インスタンス

    Raises:
        ValueError: サポートされていない手法の場合
    """
    method = method.lower()

    if method in ['simclr', 'moco']:
        temperature = kwargs.get('temperature', 0.5)
        return NTXentLoss(temperature=temperature)

    elif method == 'simsiam':
        return SimSiamLoss()

    elif method == 'barlow_twins':
        lambda_param = kwargs.get('lambda_param', 0.005)
        return BarlowTwinsLoss(lambda_param=lambda_param)

    else:
        raise ValueError(
            f"Unsupported SSL method: {method}. "
            f"Supported methods: simclr, moco, simsiam, barlow_twins"
        )
