class FixedTimeAgent:
    def __init__(self, env, green_light_time=15):
        self.env = env
        self.green_light_time = green_light_time

    def make_action(self, state, test=False):
        action = (self.env.sim_step // self.green_light_time) % 4 
        return action