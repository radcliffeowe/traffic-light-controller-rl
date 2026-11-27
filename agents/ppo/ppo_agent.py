from tqdm import tqdm
import numpy as np
import os
import json

from agents.ppo.policy_network import PPOPolicyNetwork, get_policy_actor
from agents.ppo.value_network import PPOValueNetwork, get_value_critic

from torch import device, cuda, no_grad, from_numpy, argmax, as_tensor, float32, long, bool, randperm, save, load
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict import TensorDict


class PPOAgent:
    def __init__(self, envs):
        self.envs = envs
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.metrics = {
            "episode": [], "train_reward": [], "episode_len": [],
            "policy_loss": [], "value_loss": [], "entropy": [], 
            "clip_fraction": [], "eval_episode": [], "eval_reward": []
            }
        self.metrics_dir = "agents/ppo/results"
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.num_episodes = 300
        self.trajectory_length = 240
        self.trajectory_buffer = []

        self.checkpoint_interval = 10
        self.eval_episodes = 4
        self.rand_seed_ceiling = self.num_episodes*2
        self.eval_seeds = [self.rand_seed_ceiling + 1, self.rand_seed_ceiling + 2, self.rand_seed_ceiling + 3]
        self.best_eval_reward = float("-inf")
        self.checkpoint_dir = "agents/ppo/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.policy_lr = 5e-4
        self.value_lr = 1e-3
        self.gamma = 0.95
        self.lmbda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.005

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

        self.optim = Adam([
            {"params": self.policy_network.parameters(), "lr": self.policy_lr},
            {"params": self.value_network.parameters(), "lr": self.value_lr}
        ])

        self.scheduler = LambdaLR(self.optim, lambda u: max(0.1, 1 -u / (self.num_episodes*3)))

    def load_agent(self):
        self.policy_network.load_state_dict(load("./agents/ppo/checkpoints/ppo_policy_max.pth", map_location=self.device))
        self.value_network.load_state_dict(load("./agents/ppo/checkpoints/ppo_value_max.pth", map_location=self.device))

    def eval_agent(self):
        self.policy_network.eval()
        # mean_reward, _ = test_agent(self, self.envs, self.eval_episodes, self.eval_seeds[0], None)
        total_rewards = []
        for episode in tqdm(range(self.eval_episodes)):
            env = self.envs[episode % len(self.envs)]
            truncated=False
            terminated=False
            episode_rewards = []
            state = env.reset(601)
            while not terminated and not truncated:
                if isinstance(state, tuple):
                    state = state[0]  # first observation sometimes includes info dict
                action = self.make_action(state, test=True)
                if isinstance(action, tuple):
                    action = action[0]
                state, reward, terminated, truncated, info = env.step(action)
                episode_rewards.append(reward)
            total_rewards.append(np.mean(episode_rewards))

        mean_reward = np.nanmean(total_rewards)
        self.policy_network.train()
        return mean_reward


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

    def update(self, epochs=3, minibatch_size=32):
        data = self.format_trajectory_buffer()
        update_metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "clip_fraction": []}
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
                
                update_metrics["policy_loss"].append(loss_td["loss_objective"].item())
                update_metrics["value_loss"].append(loss_td["loss_critic"].item())
                update_metrics["entropy"].append(loss_td["entropy"].item())
                if "clip_fraction" in loss_td.keys():
                    update_metrics["clip_fraction"].append(loss_td["clip_fraction"].item())

            self.scheduler.step()
        return {k: float(np.mean(v)) if v else np.nan for k,v in update_metrics.items()}

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

    def checkpoint(self, episode):
        mean_reward = self.eval_agent()
        print(f"Episode {episode} eval mean reward: {mean_reward}")
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            save(self.policy_network.state_dict(), f"{self.checkpoint_dir}/ppo_policy_max.pth")
            save(self.value_network.state_dict(), f"{self.checkpoint_dir}/ppo_value_max.pth")
        return mean_reward

    def train(self):  
        for i in tqdm(range(self.num_episodes)):
            env = self.envs[i % len(self.envs)]
            num_steps = 0
            self.trajectory_buffer.clear()
            current_trajectory_length = 0
            episode_metrics = []
            episode_rewards = []
            state, _ = env.reset(0)
            not_done = True

            while not_done:
                action, action_log_prob = self.make_action(state, False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                if terminated or truncated:
                    not_done = False

                self.store_transition(state, action, reward, next_state, action_log_prob, not not_done, terminated)
                state = next_state

                current_trajectory_length += 1
                num_steps += 1

                if current_trajectory_length % self.trajectory_length == 0:
                    # print("\nUpdating models...")
                    episode_results = self.update()
                    episode_metrics.append(episode_results)
                    current_trajectory_length = 0
                    self.trajectory_buffer.clear()

            self.metrics["episode"].append(i+1)
            episode_train_reward = float(np.mean(episode_rewards))
            self.metrics["train_reward"].append(episode_train_reward)
            print()
            print(f"Episode {i+1}, mean_reward={episode_train_reward}")
            self.metrics["episode_len"].append(len(episode_rewards))
            for key in ["policy_loss", "value_loss", "entropy", "clip_fraction"]:
                values = [s[key] for s in episode_metrics]
                self.metrics[key].append(float(np.nanmean(values)) if values else np.nan)
            if i>0 and i % self.checkpoint_interval == 0:
                # print("\nCheckpointing models...")
                eval_mean_reward = self.checkpoint(i)
                self.metrics["eval_episode"].append(i+1)
                self.metrics["eval_reward"].append(eval_mean_reward)
                print(f"**** EVAL episode {i+1}, reward: {eval_mean_reward} ****")

        with open(f"{self.metrics_dir}/training_results.json", "w") as f:
            json.dump(self.metrics, f)