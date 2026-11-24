import torch
from torch import nn
from torch.distributions import Categorical


class ReinforceAgent(nn.Module):
    """Episodic Monte Carlo policy gradient."""

    def __init__(self, env, gamma=0.95):
        super().__init__()
        observation_size = env.observation_space.shape[0]
        self.policy = self._network(observation_size, env.action_space.n)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-4)

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

    def _discounted_returns(self, rewards):
        returns = []
        discounted_return = 0.0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.append(discounted_return)
        return torch.tensor(returns[::-1], dtype=torch.float32)

    def _policy_objective(self, states, actions, advantages):
        distribution = Categorical(logits=self.policy(states))
        discounts = self.gamma ** torch.arange(len(actions), dtype=advantages.dtype)
        loss = -(discounts * distribution.log_prob(actions) * advantages).mean()
        return distribution, loss

    def update(self, observations, actions, rewards):
        returns = self._discounted_returns(rewards)
        states = torch.as_tensor(observations, dtype=torch.float32)
        action_indices = torch.as_tensor(actions, dtype=torch.long)
        distribution, policy_loss = self._policy_objective(states, action_indices, returns)

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {
            "policy_loss": policy_loss.item(),
            "entropy": distribution.entropy().mean().item(),
            "return_mean": returns.mean().item(),
        }
