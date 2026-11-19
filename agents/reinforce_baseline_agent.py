import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ReinforceBaselineAgent(nn.Module):
    """Monte Carlo policy gradient with a learned state-value baseline."""

    def __init__(self, env, gamma=0.95):
        super().__init__()
        observation_size = env.observation_space.shape[0]
        self.policy = self._network(observation_size, env.action_space.n)
        self.value = self._network(observation_size, 1)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.parameters(), "lr": 5e-4},
            {"params": self.value.parameters(), "lr": 1e-3},
        ])

    @staticmethod
    def _network(input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, output_size),
        )

    @torch.no_grad()
    def make_action(self, observation, test=True):
        logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
        if test:
            return logits.argmax().item()
        return Categorical(logits=logits).sample().item()

    def update(self, observations, actions, rewards):
        returns = []
        discounted_return = 0.0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.append(discounted_return)
        targets = torch.tensor(returns[::-1], dtype=torch.float32)
        states = torch.as_tensor(observations, dtype=torch.float32)
        action_indices = torch.as_tensor(actions, dtype=torch.long)

        distribution = Categorical(logits=self.policy(states))
        values = self.value(states).squeeze(-1)
        advantages = targets - values.detach()
        # The discounted objective also weights each decision from the episode start.
        discounts = self.gamma ** torch.arange(len(rewards), dtype=targets.dtype)
        policy_loss = -(discounts * distribution.log_prob(action_indices) * advantages).mean()
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
