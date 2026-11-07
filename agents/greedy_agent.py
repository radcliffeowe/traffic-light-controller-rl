class GreedyAgent:
    def __init__(self, env):
        self.env = env

    def make_action(self, observation, test=False):
        """
        obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue, lane_1_halted,...,lane_n_halted]
        e.g. observation: [1., 0., 1., 0.05048805, 0.1009761, 
                           0.47551954, 0.42268404, 0., 0., 0.15850651, 0.21134202, 
                           0., 0., 3., 4.]
        - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
        - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
        - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
        - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
        - ```lane_i_halted``` is the number of queued vehicles in incoming lane i
        """
        return self.env.action_space.sample()