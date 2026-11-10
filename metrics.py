from collections import defaultdict
import numpy as np
import pandas as pd

class MetricsCollector:

    def __init__(self, env):
        self.env = env
        self.max_wait = {}  # what was the longest time that a vehicle spent in a halted state
        self.total_wait = {}  # what was the total time that a vehicle spent in a halted state
        self.previous = {}
        self.queue_lane = []
        self.queue_intersection = []
        self.percentiles = {"median": 50, "p90": 90, "p95": 95, "p99": 99, "max": 100}

    def get_waiting_time_per_vehicle(self):
        for vehicle in self.env.sumo.vehicle.getIDList():
            wait_time = self.env.sumo.vehicle.getWaitingTime(vehicle)
            previous_wait = self.previous.get(vehicle, 0.0)
            self.max_wait[vehicle] = max(self.max_wait.get(vehicle, 0.0), wait_time)
            increment = wait_time - previous_wait if wait_time >= previous_wait else wait_time
            self.total_wait[vehicle] = self.total_wait.get(vehicle, 0.0) + increment
            self.previous[vehicle] = wait_time

    def get_queue_depth_metrics(self):
        for ts_id in self.env.ts_ids:
            ts = self.env.traffic_signals[ts_id]
            ts_total = 0
            for lane in ts.lanes:
                queue_depth = ts.sumo.lane.getLastStepHaltingNumber(lane)
                self.queue_lane.append(queue_depth)
                ts_total += queue_depth
            self.queue_intersection.append(ts_total)

    def record_metrics(self):
        self.get_queue_depth_metrics()
        self.get_waiting_time_per_vehicle()

    def summarize_metrics(self, samples, name):
        if len(samples) == 0:
            return {"metric": name, "n": 0}
        perc = np.percentile(samples, list(self.percentiles.values()))
        return {"metric": name, "n": len(samples), **dict(zip(self.percentiles, perc))}

    def get_summary(self):
        max_wait_times = [val for val in self.max_wait.values()]
        total_wait_times = [val for val in self.total_wait.values()]

        rows = [
            self.summarize_metrics(max_wait_times, "max_wait_per_vehicle_s"),
            self.summarize_metrics(total_wait_times, "total_wait_per_vehicle_s"),
            self.summarize_metrics(self.queue_lane, "queue_depth_per_lane"),
            self.summarize_metrics(self.queue_intersection, "queue_depth_per_intersection"),
        ]
        return pd.DataFrame(rows)