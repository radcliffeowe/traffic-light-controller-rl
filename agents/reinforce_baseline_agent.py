import torch
from torch import nn
from torch.nn import functional as F

from agents.reinforce_agent import ReinforceAgent


class ReinforceBaselineAgent(ReinforceAgent):
    """Monte Carlo policy gradient with a learned state-value baseline."""

    def __init__(self, env, gamma=0.95):
        super().__init__(env, gamma)
        self.value = self._network(env.observation_space.shape[0], 1)
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.parameters(), "lr": 5e-4},
            {"params": self.value.parameters(), "lr": 1e-3},
        ])

    def update(self, observations, actions, rewards):
        targets = self._discounted_returns(rewards)
        states = torch.as_tensor(observations, dtype=torch.float32)
        action_indices = torch.as_tensor(actions, dtype=torch.long)
        values = self.value(states).squeeze(-1)
        advantages = targets - values.detach()
        distribution, policy_loss = self._policy_objective(states, action_indices, advantages)
        value_loss = F.mse_loss(values, targets)

        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": distribution.entropy().mean().item(),
            "return_mean": targets.mean().item(),
            "value_mean": values.detach().mean().item(),
            "advantage_mean": advantages.mean().item(),
            "value_explained_variance": (
                1 - (targets - values.detach()).var(unbiased=False)
                / targets.var(unbiased=False).clamp_min(1e-8)
            ).item(),
        }
