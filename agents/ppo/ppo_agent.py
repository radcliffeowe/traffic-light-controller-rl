from tqdm import tqdm
from random import randint

from agents.ppo.policy_network import PPOPolicyNetwork
from agents.ppo.value_network import PPOValueNetwork

from torch import device, cuda, no_grad, from_numpy, argmax
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

        self.num_episodes = 10
        self.trajectory_length = 8
        self.trajectory_buffer = []

        self.checkpoint_interval = 100

        self.policy_network = PPOPolicyNetwork()
        self.value_network = PPOValueNetwork()


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
                cat_distributions = Categorical(action_softmax)
                action = cat_distributions.sample().item()
                log_probability = action_softmax[action].item()
                return action, log_probability

    def update(self):
        pass

    def store_transition(self, state, action, reward, next_state, action_log_prob, episode_done, terminated):
        self.trajectory_buffer.append([state, action, reward, next_state, action_log_prob, episode_done, terminated])

    def checkpoint(self):
        pass

    def train(self):
        num_steps = 0
        for i in tqdm(range(self.num_episodes)):
            current_trajectory_length = 0
            state, info = self.env.reset(randint(0, self.num_episodes*2))
            not_done = True
            while not_done:
                action, action_log_prob = self.make_action(state, False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    not_done = False

                self.store_transition()
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