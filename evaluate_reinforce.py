"""Compare sampled and deterministic actions from one unchanged checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sumo_rl import SumoEnvironment

from agents.reinforce_baseline_agent import ReinforceBaselineAgent
from rewards import get_mean_queued
from train_reinforce import NETWORK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[701, 702, 703])
    parser.add_argument("--num-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.num_seconds < 1:
        parser.error("num-seconds must be positive")
    for path in (args.checkpoint, args.route_file):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    torch.set_num_threads(1)
    env = SumoEnvironment(
        net_file=str(NETWORK), route_file=str(args.route_file.resolve()),
        single_agent=True, use_gui=False, num_seconds=args.num_seconds,
        reward_fn=get_mean_queued, additional_sumo_cmd="--no-step-log true",
    )
    try:
        agent = ReinforceBaselineAgent(env)
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        agent.eval()
        results = {}
        for mode in ("deterministic", "sampled"):
            episodes = []
            for seed in args.seeds:
                torch.manual_seed(seed)
                state, _ = env.reset(seed=seed)
                rewards, probabilities, requested, active = [], [], [], []
                done = False
                while not done:
                    with torch.no_grad():
                        probs = agent.policy(torch.as_tensor(state, dtype=torch.float32)).softmax(-1)
                    probabilities.append(probs.numpy())
                    active.append(int(np.argmax(state[:env.action_space.n])))
                    action = agent.make_action(state, test=mode == "deterministic")
                    requested.append(action)
                    state, reward, terminated, truncated, _ = env.step(action)
                    rewards.append(reward)
                    done = terminated or truncated
                episodes.append({
                    "seed": seed, "mean_reward": float(np.mean(rewards)),
                    "steps": len(rewards),
                    "mean_action_probabilities": np.mean(probabilities, axis=0).tolist(),
                    "requested_action_counts": np.bincount(requested, minlength=env.action_space.n).tolist(),
                    "observed_phase_counts": np.bincount(active, minlength=env.action_space.n).tolist(),
                })
            results[mode] = episodes
            print(mode, float(np.mean([episode["mean_reward"] for episode in episodes])), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
