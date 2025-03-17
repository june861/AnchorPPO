# -*- encoding: utf-8 -*-
'''
@File       :mylogger.py
@Description: Code is from https://github.com/openpsi-project/ReaLHF/blob/main/realhf/base/logging.py
@Date       :2025/03/14 11:20:46
@Author     :junweiluo
@Version    :python
'''


import logging.config
import os
from logging import WARNING, Logger, Manager, RootLogger
from typing import Literal, Optional
import colorlog
LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)d | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
LOGLEVEL = logging.INFO
SUCCESS_LEVEL = 25  # 自定义 SUCCESS 级别

# 注册 SUCCESS 级别
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# 自定义 SUCCESS 颜色
log_config = {
    "version": 1,
    "formatters": {
        "plain": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "white",
                "INFO": "white",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "blue",
                "INFO": "light_purple",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored_system": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "blue",
                "INFO": "light_green",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored_benchmark": {
            "()": colorlog.ColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "light_black",
                "INFO": "light_cyan",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
    },
    "handlers": {
        "plainHandler": {
            "class": "logging.StreamHandler",
            "level": LOGLEVEL,
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
        "benchmarkHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "colored_benchmark",
            "stream": "ext://sys.stdout",
        },
        "systemHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "colored_system",
            "stream": "ext://sys.stdout",
        },
        "coloredHandler": {
            "class": "logging.StreamHandler",
            "level": LOGLEVEL,
            "formatter": "colored",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "plain": {
            "handlers": ["plainHandler"],
            "level": LOGLEVEL,
        },
        "benchmark": {
            "handlers": ["benchmarkHandler"],
            "level": "DEBUG",
        },
        "colored": {
            "handlers": ["coloredHandler"],
            "level": LOGLEVEL,
        },
        "system": {
            "handlers": ["systemHandler"],
            "level": LOGLEVEL,
        },
    },
    "disable_existing_loggers": True,
}


class CustomLogger(logging.Logger):
    """扩展 Logger 以支持 success 方法"""
    
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, message, args, **kwargs)


def getLogger(
    name: Optional[str] = None,
    type_: Optional[Literal["plain", "benchmark", "colored", "system"]] = None,
):
    # 修复 transformer_engine 自动修改的 logging 配置
    root = RootLogger(WARNING)
    Logger.root = root
    Logger.manager = Manager(Logger.root)

    # 重新配置日志
    logging.config.dictConfig(log_config)
    if name is None:
        name = "plain"
    if type_ is None:
        type_ = "plain"
    assert type_ in ["plain", "benchmark", "colored", "system"]
    
    if name not in log_config["loggers"]:
        log_config["loggers"][name] = {
            "handlers": [f"{type_}Handler"],
            "level": LOGLEVEL,
        }
        logging.config.dictConfig(log_config)

    logger = logging.getLogger(name)
    logger.__class__ = CustomLogger  # 让 Logger 继承 CustomLogger
    return logger


if __name__ == "__main__":
    # 颜色测试
    logger = getLogger("colored", "colored")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")  # 新增 SUCCESS 级别日志
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
