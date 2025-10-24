"""
損失関数モジュール
"""
from .ssl_losses import NTXentLoss, get_ssl_loss

__all__ = ['NTXentLoss', 'get_ssl_loss']
