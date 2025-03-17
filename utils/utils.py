# -*- encoding: utf-8 -*-
'''
@File       :utils.py
@Description:
@Date       :2025/03/14 11:25:25
@Author     :junweiluo
@Version    :python
'''

import os

def check_path(path: str, logger):
    # 检查路径是否存在，不存在则递归创建
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.success(f"Path '{path}' created.")
    else:
        logger.info(f"Path '{path}' already exists.")

