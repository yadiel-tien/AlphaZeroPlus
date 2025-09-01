import logging
import os
from logging.handlers import TimedRotatingFileHandler

from utils.config import CONFIG


def get_logger(name, log_dir=CONFIG['log_dir']) -> logging.Logger:
    """生成日志logger，按名字保存"""
    log_dir = os.path.join(log_dir, name)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s')

        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, 'log'),
            when='midnight',
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger
