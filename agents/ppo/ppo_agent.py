from torch import device, cuda


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

    def make_action(self, observation, test=True):
        return self.env.action_space.sample()

    def train(self):
        pass