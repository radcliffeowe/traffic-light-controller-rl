class RandomAgent:
    def __init__(self, env):
        self.env = env

    def make_action(self, state, test=False):
        print("Making random action...")
        return self.env.action_space.sample()