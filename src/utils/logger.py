"""
ロギングユーティリティ

ファイルとコンソールへのロギングを設定します。
"""
import os
import logging
from datetime import datetime
from typing import Optional

from .constants import LOG_FORMAT, DATE_FORMAT


def setup_logger(
    name: str,
    log_dir: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    ロガーを設定（ファイルとコンソールハンドラー付き）

    Args:
        name: ロガー名
        log_dir: ログディレクトリ
        log_file: ログファイル名（Noneの場合は自動生成）
        level: ログレベル

    Returns:
        設定済みロガー
    """
    os.makedirs(log_dir, exist_ok=True)

    # ログファイル名を決定
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"

    log_path = os.path.join(log_dir, log_file)

    # ロガーを作成（既存のロガーを取得）
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ハンドラーがすでに存在する場合はスキップ（重複防止）
    if logger.handlers:
        return logger

    # フォーマッターを作成
    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    # ファイルハンドラー
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ハンドラーを追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 親ロガーへの伝播を防止（重複ログ防止）
    logger.propagate = False

    return logger
