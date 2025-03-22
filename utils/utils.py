# -*- encoding: utf-8 -*-
'''
@File       :utils.py
@Description:
@Date       :2025/03/14 11:25:25
@Author     :junweiluo
@Version    :python
'''

import time
import wandb
import os

def check_path(path: str, logger):
    # 检查路径是否存在，不存在则递归创建
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.success(f"Path '{path}' created.")
    else:
        logger.info(f"Path '{path}' already exists.")


def start_wandb_proj(args):
    if args.project_name == None:
        args.project_name = 'atari'
    if args.algo == 'appo':
        all_act_sampled = 'all' if args.use_all else 'two'
        run_name = f'atari-{args.env_id}-{args.algo}_{all_act_sampled}-seed{args.seed}-epoch{args.update_epochs}-{int(time.time())}'
        group_name = f'atari-{args.env_id}-{args.algo}_{all_act_sampled}-epoch{args.update_epochs}'
    else:
        run_name = f'atari-{args.env_id}-{args.algo}-seed{args.seed}-epoch{args.update_epochs}-{int(time.time())}'
        group_name = f'atari-{args.env_id}-{args.algo}-epoch{args.update_epochs}'
    wandb.init(
        project=args.project_name,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        group=group_name,
        monitor_gym=True,
        save_code=True,
    )