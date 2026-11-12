from torch import device, cuda


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

        self.lr = 3e-4
        self.max_grad_norm = 1.0
        self.batch_size = 1000
        self.sub_batch_size = 64
        self.num_epochs = 10
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95
        self.entropy_eps = 1e-4


    def make_action(self, observation, test=True):
        """
        obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
        e.g. observation: [1., 0., 0., 0., 1., 0.05048805, 0.1009761, 
                           0.47551954, 0.42268404, 0., 0., 0.15850651, 0.21134202]
        - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase, i.e. [1., 0., 0., 0.]
        - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
        - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
        - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
        """
        return self.env.action_space.sample()

    def train(self):
        pass