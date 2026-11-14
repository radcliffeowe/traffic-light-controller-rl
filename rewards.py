from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np

def get_mean_queued(ts: TrafficSignal) -> float:
    """Returns the weighted mean of vehicles queued."""
    lane_capacities =  [ts.lanes_length[lane] / (ts.MIN_GAP + ts.sumo.lane.getLastStepLength(lane)) for lane in ts.lanes]
    lane_queues = ts.get_lanes_queue()
    return -np.average(lane_queues, weights=lane_capacities)