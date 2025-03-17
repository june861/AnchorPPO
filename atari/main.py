import argparse
import time
import sys
import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange
from utils import check_path
from agent import Agent
from buffer import Buffer
from trainer import Trainer
from config import get_config


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--use_resnet', type=bool, default=True)
#     parser.add_argument('--use_cuda', type=bool, default=True)
#     parser.add_argument('--torch_deterministic', type=bool, default=True)
#     parser.add_argument('--total_time_steps', type=int, default=int(1e7))
#     parser.add_argument('--learning_rate', type=float, default=2.5e-4)
#     parser.add_argument('--learning_rate_decay', type=bool, default=True)
#     parser.add_argument('--num_envs', type=int, default=8)
#     parser.add_argument('--num_steps', type=int, default=128)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--gae_lambda', type=float, default=0.95)
#     parser.add_argument('--mini_batches', type=int, default=4)
#     parser.add_argument('--update_epochs', type=int, default=4)
#     parser.add_argument('--advantage_normalization', type=bool, default=True)
#     parser.add_argument('--clip_value_loss', type=bool, default=True)
#     parser.add_argument('--c_1', type=float, default=0.5)
#     parser.add_argument('--c_2', type=float, default=0.01)
#     parser.add_argument('--max_grad_norm', type=float, default=0.5)
#     parser.add_argument('--epsilon', type=float, default=0.2)
#     args = parser.parse_args()
#     args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
#     args.batch_size = int(args.num_envs * args.num_steps)
#     args.minibatch_size = int(args.batch_size // args.mini_batches)
#     args.num_updates = int(args.total_time_steps // args.batch_size)
#     return args


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def compute_advantages(rewards, flags, values, next_value, args):
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    adv = torch.zeros(args.num_envs).to(args.device)
    for i in reversed(range(args.num_steps)):
        returns = rewards[i] + args.gamma * flags[i] * next_value
        delta = returns - values[i]
        adv = delta + args.gamma * args.gae_lambda * flags[i] * adv
        advantages[i] = adv
        next_value = values[i]
    return advantages


def train():

    args = get_config()

    # args.algo = algo
    network = 'resnet' if args.use_resnet else 'cnn'
    run_name = args.algo + '_' + str(args.epsilon) + '_' + network + '_seed_' + str(args.seed)
    print('[algorithm:', args.algo + ']', '[env:', args.env_id + ']', '[seed:', str(args.seed) + ']')

    
    
    # path_string = str(args.env_id)[:-14] + '/' + run_name
    check_path(args.run_dir, args.logger)
    writer = SummaryWriter(args.run_dir)
    writer.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )


    if args.use_wandb:
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

    # Initialize environments
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # State space and action space
    observation_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n

    # Random seed
    if args.torch_deterministic:
        numpy_rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        state, _ = envs.reset(seed=args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    else:
        numpy_rng = np.random.default_rng()
        state, _ = envs.reset()

    # Initialize agent and optimizer
    agent = Agent(num_actions, args.use_resnet).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    trainer = Trainer(args, agent, optimizer, writer)

    # Initialize buffer
    rollout_buffer = Buffer(
        num_steps = args.num_steps, 
        num_envs = args.num_envs, 
        observation_shape = observation_shape, 
        device = args.device, 
        discrete_action_n = num_actions,
    )
    global_step = 0
    start_time = time.time()

    # This is for plotting
    episodic_returns = []
    update_index = 0
    for update in trange(1, args.num_updates + 1):

        # Linear decay of learning rate
        if args.learning_rate_decay:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(args.num_steps):
            global_step += args.num_envs

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value, total_logits = agent.get_action_and_value(
                    torch.from_numpy(state).to(args.device).float()
                )
            action = action.cpu().numpy()

            # Update the environments
            next_state, reward, terminated, truncated, all_info = envs.step(action)

            # Save data
            flag = 1.0 - np.logical_or(terminated, truncated)
            log_prob = log_prob.cpu().numpy()
            value = value.cpu().numpy()
            total_logits = total_logits.cpu().numpy()
            rollout_buffer.push(state, action, reward, flag, log_prob, value, total_logits)
            state = next_state

            if 'final_info' in all_info:
                for info in all_info['final_info']:
                    if info and 'episode' in info:
                        writer.add_scalar('charts/episodic_return', info['episode']['r'], global_step)
                        # args.logger.info(f"global_step is {global_step}, episodic return is {info['episode']['r']}")
                        if update // 15 == update_index:
                            episodic_returns.append(info['episode']['r'])
                        else:
                            writer.add_scalar(
                                'charts/average_return', np.mean(episodic_returns), update_index + 1
                            )
                            episodic_returns.clear()
                            episodic_returns.append(info['episode']['r'])
                            update_index += 1
                            

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        states, actions, rewards, flags, log_probs, values, old_logits = rollout_buffer.get()

        # Use GAE technique to estimate the advantage
        with torch.no_grad():
            next_value = agent.get_value(torch.from_numpy(next_state).to(args.device).float())
            advantages = compute_advantages(rewards, flags, values, next_value, args)
            returns = advantages + values

        # Flatten each batch
        b_states = states.reshape(-1, *observation_shape)
        b_actions = actions.reshape(-1)
        b_log_probs = log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_old_logits = old_logits.reshape((-1, num_actions)) # shape is (num_steps* num_envs, num_actions)

        # Update the policy network and value network
        trainer.train(
            numpy_rng = numpy_rng, 
            global_step = global_step, 
            b_obs =  b_states, 
            b_actions = b_actions, 
            b_log_probs = b_log_probs, 
            b_advantages = b_advantages, 
            b_returns = b_returns, 
            b_values = b_values, 
            b_old_logits = b_old_logits,
        )

        explained_var = (
            np.nan if torch.var(b_returns) == 0 else 1 - torch.var(b_returns - b_values) / torch.var(b_returns)
        )
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar('losses/explained_variance', explained_var, global_step)

    envs.close()
    writer.close()


# def main(algo):
#     for env_id in [
#         'Assault',
#         'Asterix',
#         'BeamRider',
#         'SpaceInvaders',
#     ]:
#         for seed in [1, 2, 3]:
#             train(algo, env_id + 'NoFrameskip-v4', seed)


if __name__ == '__main__':
    train()
