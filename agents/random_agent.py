class RandomAgent:
    def __init__(self, env):
        self.env = env

    def make_action(self, state, test=False):
        return self.env.action_space.sample()