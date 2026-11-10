import numpy as np

class GreedyAgent:
    """
    Agent that always picks the phase that is the most congested at the
    current time step.
    """
    def __init__(self, env):
        self.env = env

    def index_lanes_to_phases(self, ts) -> list:
        lane_to_index_map = {lane: i for i, lane in enumerate(ts.lanes)}
        controlled_links = ts.sumo.trafficlight.getControlledLinks(ts.id)
        phase_lanes = []
        for phase in ts.green_phases:
            lanes = set()
            for i, state_repr in enumerate(phase.state):
                if state_repr in ("G", "g") and controlled_links[i]:
                    lanes.add(lane_to_index_map[controlled_links[i][0][0]])
            phase_lanes.append(sorted(lanes))
        return phase_lanes

    def make_action(self, observation: np.ndarray, test=False):
        """
        obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
        e.g. observation: [1., 0., 1., 0.05048805, 0.1009761, 
                           0.47551954, 0.42268404, 0., 0., 0.15850651, 0.21134202]
        - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
        - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
        - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
        - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
        """
        self.ts = self.env.traffic_signals[self.env.ts_ids[0]]
        self.num_phases = self.ts.num_green_phases
        self.num_lanes = len(self.ts.lanes)
        self.lanes_to_phases_index = self.index_lanes_to_phases(self.ts)
        self.min_green_position = self.num_phases
        self.queue_slice = slice(self.num_phases + 1 + self.num_lanes, self.num_phases + 1 + 2*self.num_lanes)

        current = int(np.argmax(observation[:self.num_phases]))
        if observation[self.min_green_position] == 0:
            return current

        queue = observation[self.queue_slice]
        scores = [queue[index].sum() for index in self.lanes_to_phases_index]

        best_score = max(scores)
        if scores[current] == best_score:
            return current
        return int(np.argmax(scores))