from tqdm import tqdm
from random import randint
import numpy as np
import os

from agents.ppo.policy_network import PPOPolicyNetwork, get_policy_actor
from agents.ppo.value_network import PPOValueNetwork, get_value_critic
from test_agent import test_agent

from torch import device, cuda, no_grad, from_numpy, argmax, as_tensor, float32, long, bool, randperm
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.distributions import Categorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict import TensorDict


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

        self.num_episodes = 5
        self.trajectory_length = 8
        self.trajectory_buffer = []

        self.checkpoint_interval = 2
        self.eval_episodes = 3
        self.rand_seed_ceiling = self.num_episodes*2
        self.eval_seeds = [self.rand_seed_ceiling + 1, self.rand_seed_ceiling + 2, self.rand_seed_ceiling + 3]
        self.best_eval_reward = float("-inf")
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.lr = 1e-4
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01

        self.policy_network = PPOPolicyNetwork().to(self.device)
        self.value_network = PPOValueNetwork().to(self.device)

        self.policy_actor = get_policy_actor(self.policy_network)
        self.value_critic = get_value_critic(self.value_network)
        self.advantage_module = GAE(
            gamma=self.gamma, 
            lmbda=self.lmbda, 
            value_network=self.value_critic, 
            average_gae=True
            )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_actor, 
            critic_network=self.value_critic, 
            clip_epsilon=self.clip_epsilon, 
            entropy_coeff=self.entropy_coeff
            )

        self.optim = Adam(params=self.loss_module.parameters(), lr=self.lr)

    def eval_agent(self):
        self.policy_network.eval()
        test_agent(self, self.env, self.eval_episodes, self.eval_seeds[0], self.checkpoint_dir)

    def make_action(self, observation, test=True):
        """
        obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
        e.g. observation: [1., 0., 0., 0., 1., 0.05048805, 0.1009761, 
                           0.47551954, 0.42268404, 0., 0., 0.15850651, 0.21134202]
        - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase, i.e. [1., 0., 0., 0.]
        - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
        - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
        - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
        
        Returns the action and the log probability of that action from the softmax output.
        """
        torch_observation = from_numpy(observation).float().to(self.device)
        with no_grad():
            action_softmax = self.policy_network(torch_observation)
            if test:
                # deterministic
                return argmax(action_softmax).item(), None
            else:
                # stochastic
                cat_distributions = Categorical(probs=action_softmax)
                action = cat_distributions.sample()
                log_probability = cat_distributions.log_prob(action).item()
                return action.item(), log_probability

    def update(self, epochs=4, minibatch_size=8):
        data = self.format_trajectory_buffer()

        for _ in range(epochs):
            with no_grad():
                self.advantage_module(data)

            permutation = randperm(data.shape[0], device=data.device)
            for index in permutation.split(minibatch_size):
                minibatch = data[index]
                loss_td = self.loss_module(minibatch)
                loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]

                self.optim.zero_grad()
                loss.backward()
                clip_grad_norm_(self.loss_module.parameters(), 1.0)
                self.optim.step()

    def format_trajectory_buffer(self):
        state, action, reward, next_state, action_log_prob, episode_done, terminated = zip(*self.trajectory_buffer)
            
        state = as_tensor(np.stack(state), dtype=float32)
        action = as_tensor(action, dtype=long)
        reward = as_tensor(reward, dtype=float32).unsqueeze(-1)
        next_state = as_tensor(np.stack(next_state), dtype=float32)
        action_log_prob = as_tensor(action_log_prob, dtype=float32)
        episode_done = as_tensor(episode_done, dtype=bool).unsqueeze(-1)
        terminated = as_tensor(terminated, dtype=bool).unsqueeze(-1)

        return TensorDict(
            {
                "state": state,
                "action": action,
                "action_log_prob": action_log_prob,
                "next": {
                    "state": next_state,
                    "reward": reward,
                    "done": episode_done,
                    "terminated": terminated
                },
            },
            batch_size=[len(self.trajectory_buffer)],
            device=self.device
        )

    def store_transition(self, state, action, reward, next_state, action_log_prob, episode_done, terminated):
        self.trajectory_buffer.append([state, action, reward, next_state, action_log_prob, episode_done, terminated])

    def checkpoint(self):
        pass

    def train(self):
        num_steps = 0
        for i in tqdm(range(self.num_episodes)):
            self.trajectory_buffer.clear()
            current_trajectory_length = 0
            state, info = self.env.reset(randint(0, self.rand_seed_ceiling))
            not_done = True
            while not_done:
                action, action_log_prob = self.make_action(state, False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    not_done = False

                self.store_transition(state, action, reward, next_state, action_log_prob, not not_done, terminated)
                state = next_state

                current_trajectory_length += 1
                num_steps += 1

                if current_trajectory_length % self.trajectory_length == 0:
                    # print("\nUpdating models...")
                    self.update()
                    current_trajectory_length = 0
                    self.trajectory_buffer.clear()

                if num_steps % self.checkpoint_interval == 0:
                    # print("\nCheckpointing models...")
                    self.checkpoint()