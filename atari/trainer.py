import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils import monitor_gradient

class Trainer:
    def __init__(self, args, agent, optimizer, writer):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        # junweiluo
        self.batch_index = 0
        self.pow_epsilon = self.args.num_actions - 1 if self.args.algo == 'appo-all' else 1

    def train(self, numpy_rng, global_step, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values, b_old_logits):
        b_index = np.arange(self.args.batch_size)

        for epoch in range(self.args.update_epochs):
            numpy_rng.shuffle(b_index)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, new_entropy, new_value, new_logits = self.agent.get_action_and_value(
                    b_obs[mb_index], b_actions[mb_index]
                )

                # Probability ratio
                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()

                # Advantage normalization
                mb_advantages = b_advantages[mb_index]
                # 暂时取消掉这个if判断
                # if self.args.advantage_normalization:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                policy_loss, ratios, ratio1, ratio2 = self.compute_policy_loss(ratios, mb_advantages, b_old_logits[mb_index], new_logits, b_actions[mb_index])

                # Value loss
                value_loss = self.compute_value_loss(new_value, b_returns[mb_index], b_values[mb_index])
                # Policy entropy
                entropy_loss = new_entropy.mean()
                # Total loss
                loss = policy_loss + value_loss * self.args.c_1 - entropy_loss * self.args.c_2

                self.writer.add_scalar('charts/ratio_deviation', (ratios - 1).mean(), self.batch_index)
                self.writer.add_scalar('charts/ratio1_deviation', (ratio1 - 1).mean(), self.batch_index)
                self.writer.add_scalar('charts/ratio2_deviation', (ratio2 - 1).mean(), self.batch_index)

                
                self.writer.add_scalar('imp_weights/ratio', ratios.mean(), self.batch_index)
                self.writer.add_scalar('imp_weights/ratio1', ratio1.mean(), self.batch_index)
                self.writer.add_scalar('imp_weights/ratio2', ratio2.mean(), self.batch_index)
                # junweiluo: log adv relevant data
                # indice_larger_0 = np.where(b_returns[mb_index].detach().cpu().numpy() > 0)[0]
                # indice_smaller_0 = np.where(b_returns[mb_index].detach().cpu().numpy() < 0)[0]
                # adv_larger0_ratio1 = np.mean(ratio1.detach().cpu().numpy()[indice_larger_0])
                # adv_larger0_ratio2 = np.mean(ratio2.detach().cpu().numpy()[indice_larger_0])
                # adv_larger0_ratio = np.mean(ratios.detach().cpu().numpy()[indice_larger_0])
                # adv_smaller0_ratio1 = np.mean(ratio1.detach().cpu().numpy()[indice_smaller_0])
                # adv_smaller0_ratio2 = np.mean(ratio2.detach().cpu().numpy()[indice_smaller_0])
                # adv_smaller0_ratio = np.mean(ratios.detach().cpu().numpy()[indice_smaller_0])
                # self.writer.add_scalar('adv/adv_larger0_ratio1', adv_larger0_ratio1, self.batch_index)
                # self.writer.add_scalar('adv/adv_larger0_ratio2', adv_larger0_ratio2, self.batch_index)
                # self.writer.add_scalar('adv/adv_larger0_ratio', adv_larger0_ratio, self.batch_index)
                # self.writer.add_scalar('adv/adv_smaller0_ratio1', adv_smaller0_ratio1, self.batch_index)
                # self.writer.add_scalar('adv/adv_smaller0_ratio2', adv_smaller0_ratio2, self.batch_index)
                # self.writer.add_scalar('adv/adv_smaller0_ratio', adv_smaller0_ratio, self.batch_index)
                
                min_ratio, max_ratio = np.min(ratios.detach().cpu().numpy()), np.max(ratios.detach().cpu().numpy())
                min_ratio1, max_ratio1 = np.min(ratio1.detach().cpu().numpy()), np.max(ratio1.detach().cpu().numpy())
                min_ratio2, max_ratio2 = np.min(ratio2.detach().cpu().numpy()), np.max(ratio2.detach().cpu().numpy())
                self.writer.add_scalar('imp_weight/min_ratio', min_ratio, self.batch_index)
                self.writer.add_scalar('imp_weight/max_ratio', max_ratio, self.batch_index)
                self.writer.add_scalar('imp_weight/min_ratio1', min_ratio1, self.batch_index)
                self.writer.add_scalar('imp_weight/max_ratio1', max_ratio1, self.batch_index)
                self.writer.add_scalar('imp_weight/min_ratio2', min_ratio2, self.batch_index)
                self.writer.add_scalar('imp_weight/max_ratio2', max_ratio2, self.batch_index)
                
                if start == 0 and epoch !=0 :
                    # 计算KL
                    kl_divs = torch.distributions.kl.kl_divergence(
                        Categorical(logits=b_old_logits[mb_index]), 
                        Categorical(logits=new_logits),
                    ).mean()
                    self.writer.add_scalar('losses/kl_div', kl_divs.item())
                
                # Update network parameters
                self.optimizer.zero_grad()
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.writer.add_scalar('losses/grad_norm', grad, self.batch_index)
                self.optimizer.step()
                
                
                self.batch_index += 1

        
        self.writer.add_scalar('losses/policy_loss', policy_loss.item(), self.batch_index)
        self.writer.add_scalar('losses/value_loss', value_loss.item(), self.batch_index)
        self.writer.add_scalar('losses/entropy', entropy_loss.item(), self.batch_index)

    def compute_value_loss(self, new_value, mb_returns, mb_values):
        """
        Compute value loss
        """
        new_value = new_value.view(-1)
        # if self.args.clip_value_loss:
        value_loss_un_clipped = (new_value - mb_returns) ** 2
        value_clipped = mb_values + torch.clamp(new_value - mb_values, -self.args.epsilon, self.args.epsilon)
        value_loss_clipped = (value_clipped - mb_returns) ** 2
        value_loss_max = torch.max(value_loss_un_clipped, value_loss_clipped)
        value_loss = 0.5 * value_loss_max.mean()
        # else:
        #     value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

        return value_loss


    def compute_policy_loss(self, ratios, mb_advantages, b_old_logits, new_logits, b_actions):
        """
        Compute the policy loss without using if-else statements
        """

        # 定义不同算法对应的计算方法
        algo_map = {
            'ppo': self._compute_ppo_loss,
            'appo-pow': self._compute_appo_pow_loss,
            'appo-all': self._compute_appo_all_two_loss,
            'appo-two': self._compute_appo_all_two_loss,
            'spo': self._compute_spo_loss
        }

        # 直接通过字典映射调用对应函数
        policy_func = algo_map.get(self.args.algo, self._algo_not_implemented)
        return policy_func(ratios, mb_advantages, b_old_logits, new_logits, b_actions)


    # ========== PPO Loss ==========
    def _compute_ppo_loss(self, ratios, mb_advantages, *args):
        policy_loss_1 = mb_advantages * ratios
        policy_loss_2 = mb_advantages * torch.clamp(
            ratios, 1 - self.args.epsilon, 1 + self.args.epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        return policy_loss, ratios, ratios, ratios


    # ========== APPO-Pow Loss ==========
    def _compute_appo_pow_loss(self, ratios, mb_advantages, *args):
        ratio_1 = ratios.detach()
        ratios = ratios * ratio_1
        policy_loss_1 = mb_advantages * ratios
        policy_loss_2 = mb_advantages * torch.clamp(
            ratios, 1 - self.args.epsilon, 1 + self.args.epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        return policy_loss, ratios, ratio_1, ratio_1

    
    def _compute_appo_all_two_loss(self, ratios, mb_advantages, b_old_logits, new_logits, b_actions):
        ratio1 = ratios.detach()
        num_alter_actions = b_old_logits.shape[1] - 1 if self.args.algo == 'appo-all' else 1
        mask_ = torch.ones_like(b_old_logits)
        mask_[torch.arange(b_old_logits.shape[0]), b_actions] = 0.0
        selected_indice = torch.multinomial(mask_, num_samples=num_alter_actions).squeeze()
        selected_indice_0 = torch.arange(b_old_logits.shape[0])
        if num_alter_actions > 1:
            selected_indice_0 = selected_indice_0.unsqueeze(1).expand(-1, num_alter_actions)
        old_logprobs = b_old_logits[selected_indice_0, selected_indice]
        new_logprobs = new_logits[selected_indice_0, selected_indice]
        log_ratio2 = new_logprobs - old_logprobs
        if len(log_ratio2.shape) == 1:
            log_ratio2 = log_ratio2.unsqueeze(1)
        # 乘积形式
        ratio2 = torch.sum(log_ratio2, dim=1).exp()
        # raw_ratio2 = torch.pow(raw_ratio2,  1 / num_alter_actions)
        ratio2 = torch.clamp(ratio2, 1 - self.args.epsilon_2, 1 + self.args.epsilon_2)
        ratios = ratios * ratio2
        # 取均值的形式
        # ratio2 = torch.sum(log_ratio2.exp(), dim = 1).detach()
        # ratios = (ratios + ratio2) / (num_alter_actions + 1)
        policy_loss_1 = mb_advantages * ratios
        policy_loss_2 = mb_advantages * torch.clamp(
            ratios, 1 - self.args.epsilon, 1 + self.args.epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        return policy_loss, ratios, ratio1, ratio2


    # ========== SPO Loss ==========
    def _compute_spo_loss(self, ratios, mb_advantages, *args):
        policy_loss = -(
            mb_advantages * ratios -
            torch.abs(mb_advantages) * torch.pow(ratios - 1, 2) / (2 * self.args.epsilon)
        ).mean()
        return policy_loss, ratios, ratios, ratios


    # ========== 未实现算法处理 ==========
    def _algo_not_implemented(self, *args):
        self.args.logger.error(f'Not Implemented for such algo: {self.args.algo}')
        raise NotImplementedError(f'Algorithm {self.args.algo} is not implemented')
