import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from agents.reinforce_agent import ReinforceAgent


class ReinforceTests(unittest.TestCase):
    def make_agent(self, learning_rate=0.0):
        env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(2,)),
            action_space=SimpleNamespace(n=2),
        )
        agent = ReinforceAgent(env, gamma=0.5)
        agent.policy = nn.Linear(2, 2)
        with torch.no_grad():
            agent.policy.weight.zero_()
            agent.policy.bias.zero_()
        agent.optimizer = torch.optim.SGD(agent.policy.parameters(), lr=learning_rate)
        return agent

    def test_return_weighted_loss_and_gradient(self):
        agent = self.make_agent()
        rewards = [-1, -2, -3]
        returns = np.array([-2.75, -3.5, -3.0])
        weights = np.array([1.0, 0.5, 0.25])
        actions = np.array([0, 1, 0])
        result = agent.update(np.zeros((3, 2)), actions, rewards)

        self.assertAlmostEqual(result["return_mean"], returns.mean(), places=6)
        self.assertAlmostEqual(
            result["policy_loss"], math.log(2) * np.mean(weights * returns), places=6
        )
        expected_gradient = [
            -np.mean(weights * returns * ((actions == action).astype(float) - 0.5))
            for action in range(2)
        ]
        np.testing.assert_allclose(agent.policy.bias.grad.numpy(), expected_gradient, atol=1e-6)
        self.assertFalse(hasattr(agent, "value"))
        self.assertNotIn("value_loss", result)

    def test_return_sign_changes_action_probability(self):
        for reward, direction in ((0.25, 1), (-0.25, -1)):
            with self.subTest(reward=reward):
                agent = self.make_agent(learning_rate=0.1)
                agent.update(np.zeros((1, 2)), [0], [reward])
                with torch.no_grad():
                    probability = agent.policy(torch.zeros(2)).softmax(-1)[0].item()
                self.assertGreater(direction * (probability - 0.5), 0)

    def test_zero_returns_leave_policy_unchanged(self):
        agent = self.make_agent(learning_rate=0.1)
        before = {name: value.detach().clone() for name, value in agent.policy.named_parameters()}
        agent.update(np.zeros((2, 2)), [0, 1], [0.0, 0.0])
        for name, value in agent.policy.named_parameters():
            torch.testing.assert_close(value, before[name])


if __name__ == "__main__":
    unittest.main()
