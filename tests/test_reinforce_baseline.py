import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from agents.reinforce_baseline_agent import ReinforceBaselineAgent


class ReinforceUpdateTests(unittest.TestCase):
    def make_agent(self, baseline=0.0, learning_rate=0.0):
        env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(2,)),
            action_space=SimpleNamespace(n=2),
        )
        agent = ReinforceBaselineAgent(env, gamma=0.5)
        agent.policy = nn.Linear(2, 2)
        agent.value = nn.Linear(2, 1)
        with torch.no_grad():
            agent.policy.weight.zero_()
            agent.policy.bias.zero_()
            agent.value.weight.zero_()
            agent.value.bias.fill_(baseline)
        agent.optimizer = torch.optim.SGD(agent.parameters(), lr=learning_rate)
        return agent

    def test_returns_losses_and_detached_baseline_gradients(self):
        # Rewards [-1, -2, -3], gamma 0.5 give returns [-2.75, -3.5, -3].
        # Baseline -3 gives advantages [0.25, -0.5, 0].
        returns = np.array([-2.75, -3.5, -3.0])
        advantages = np.array([0.25, -0.5, 0.0])
        actions = np.array([0, 1, 0])
        weights = np.array([1.0, 0.5, 0.25])
        agent = self.make_agent(baseline=-3.0)
        result = agent.update(np.zeros((3, 2)), actions, [-1, -2, -3])
        self.assertAlmostEqual(result["return_mean"], returns.mean(), places=6)
        self.assertAlmostEqual(
            result["policy_loss"], math.log(2) * np.mean(weights * advantages), places=6
        )
        self.assertAlmostEqual(result["value_loss"], np.mean(advantages ** 2), places=6)

        expected_policy_gradient = []
        for action in range(2):
            score_gradient = (actions == action).astype(float) - 0.5
            expected_policy_gradient.append(-np.mean(weights * advantages * score_gradient))
        np.testing.assert_allclose(
            agent.policy.bias.grad.numpy(), expected_policy_gradient, atol=1e-6
        )
        # Only value MSE contributes here. Policy loss must not train the baseline.
        expected_value_gradient = 2 * np.mean(-3.0 - returns)
        self.assertAlmostEqual(
            agent.value.bias.grad.item(), expected_value_gradient, places=6
        )

    def test_advantage_sign_changes_chosen_action_probability(self):
        for reward, should_increase in ((0.25, True), (-0.25, False)):
            with self.subTest(reward=reward):
                agent = self.make_agent(learning_rate=0.1)
                state = np.zeros((1, 2))
                agent.update(state, [0], [reward])
                with torch.no_grad():
                    probability = agent.policy(torch.zeros(2)).softmax(-1)[0].item()
                if should_increase:
                    self.assertGreater(probability, 0.5)
                else:
                    self.assertLess(probability, 0.5)

    def test_zero_advantage_leaves_policy_unchanged(self):
        agent = self.make_agent(learning_rate=0.1)
        before = {name: parameter.detach().clone() for name, parameter in agent.policy.named_parameters()}
        agent.update(np.zeros((1, 2)), [0], [0.0])
        for name, parameter in agent.policy.named_parameters():
            torch.testing.assert_close(parameter, before[name])


if __name__ == "__main__":
    unittest.main()
