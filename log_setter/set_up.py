import logging
from logging import FileHandler, Formatter, getLogger
import platform
import subprocess


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


def get_git_info(logger: logging.Logger) -> logging.Logger:
    """
    Gitの情報を取得し, 実験者環境をlogファイルに記録する関数

    Parameters
    ------------
    logger: logging.Logger
        記録するlogデータ

    Returens
    ------------
    logger: logging.Logger
        記録するlogデータ
    """

    # commit idを取得する
    git_info = "commit id: "
    git_info += subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()

    # 実行者名を取得する
    git_name = "username: "
    git_name += subprocess.check_output(
        ["git", "config", "user.name"]
    ).decode().strip()

    # 上で取得したデータをlogに記録する
    logger.info(git_info)
    logger.info(git_name)

    return logger


def get_os_info(logger: logging.Logger) -> logging.Logger:
    """
    OSの情報を取得し, 実験者環境をlogファイルに記録する関数

    Parameters
    ------------
    logger: logging.Logger
        記録するlogデータ

    Returens
    ------------
    logger: logging.Logger
        記録するlogデータ
    """

    # OSのスペックをまとめる
    os_info = "\n\n"
    spec = [
        f"\tOS: {platform.system()} {platform.release()}\n",
        f"\tProcessor: {platform.processor()}\n",
        f"\tMachine: {platform.machine()}\n",
        f"\tNode: {platform.node()}\n",
        f"\tPython Version: {platform.python_version()}\n"
    ]
    os_info += "".join(spec)

    # 上でまとめたスペックをlogに記録する
    logger.info(f"OS information: {os_info}")

    return logger


def get_pip_list(logger: logging.Logger) -> logging.Logger:
    """
    使用ライブラリの情報を取得し, Pipリストをlogファイルに記録する関数

    Parameters
    ------------
    logger: logging.Logger
        記録するlogデータ

    Returens
    ------------
    logger: logging.Logger
        記録するlogデータ
    """

    # Pipリストを取得してまとめる
    pip_list = "pip3 list:\n\n"
    pip_list += subprocess.check_output(
        ["pip3", "list"]
    ).decode().strip()
    pip_list += "\n"

    # 上でまとめたPipリストをlogに記録する
    logger.info(pip_list)

    return logger


def set_logging(filename: str) -> logging.Logger:
    """
    使用ライブラリの情報を取得し, Pipリストをlogファイルに記録する関数

    Parameters
    ------------
    logger: logging.Logger
        記録するlogデータ

    Returens
    ------------
    logger: logging.Logger
        記録するlogデータ
    """

    # logの初期化を行う
    logger = get_logger(filename)

    # 環境をlogに記録する
    logger = get_git_info(logger)
    logger = get_os_info(logger)
    logger = get_pip_list(logger)

    return logger
