import logging
from logging import FileHandler, Formatter, getLogger


def get_logger(filename: str) -> logging.Logger:
    """
    log の初期化を行う関数

    Parameters
    ----------
    filename: str
        logを保存するファイル名

    Returns
    ----------
    logger: logging.Logger
        実験の結果を記録する log データ
    """
    # logger インスタンスを作成し, log を初期化する
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)

    # handler と formatter を作成する
    log_formatter = "%(asctime)s - %(levelname)s - %(message)s"
    handler = FileHandler(filename, mode="w")
    formatter = Formatter(log_formatter)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
