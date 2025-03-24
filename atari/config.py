import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import getLogger
from prettytable import PrettyTable


def get_config():
    parser = argparse.ArgumentParser()
    # BreakoutNoFrameskip-v4 AssaultNoFrameskip-v4
    parser.add_argument('--env_id', type=str, default="AssaultNoFrameskip-v4")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--algo', type=str, default='appo-all', choices=['ppo', 'appo-all', 'appo-two', 'spo', 'appo-pow'])
    
    parser.add_argument('--use_resnet', type=bool, default=False)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--torch_deterministic', type=bool, default=True)
    parser.add_argument('--total_time_steps', type=int, default=int(1e7))
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--learning_rate_decay', type=bool, default=True)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--mini_batches', type=int, default=4)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--advantage_normalization', type=bool, default=True)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--epsilon_2', type=float, default=0.2)

    parser.add_argument('--use_wandb', action='store_false', default=True)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--use_all', action='store_false', default=True)
    parser.add_argument('--run_dir', type=str, default=os.path.join(os.getcwd(),'runs', 'atari'))

    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.mini_batches)
    args.num_updates = int(args.total_time_steps // args.batch_size)
    
    # logger 
    args.logger = getLogger(f"{args.env_id}","colored")
    args.logger.success(f"all arguments are configured successfully!")


    # 表格输出配置项
    config_table = PrettyTable()
    config_table.field_names = ["Parameter", "Value"]

    # 添加配置项到表格
    for arg in vars(args):
        config_table.add_row([arg, getattr(args, arg)])

    # 打印配置表格
    args.logger.info(f'\n{config_table}')

    return args