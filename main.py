import argparse

from test_agent import test_agent
from observations import LaneQueueObservationFunction

import gymnasium as gym
import sumo_rl

def parse():
    parser = argparse.ArgumentParser(description="Sumo RL Agents")
    parser.add_argument('--agent_name', required=True)
    parser.add_argument('--test_agent', action='store_true')
    parser.add_argument('--net_file')
    parser.add_argument('--route_file')
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--use_gui', action='store_true')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args):
    if args.test_agent:
        env = sumo_rl.SumoEnvironment(
                net_file=args.net_file,
                route_file=args.route_file,
                out_csv_name=args.output_path,
                single_agent=True,
                use_gui=args.use_gui,
                num_seconds=1000,
                reward_fn="queue"
                )

        try:
            if args.agent_name == "random":
                from agents.random_agent import RandomAgent
                agent = RandomAgent(env)
            elif args.agent_name == "greedy":
                from agents.greedy_agent import GreedyAgent
                agent = GreedyAgent(env)
            else:
                print("Agent not found, exiting...")
                raise Exception("Agent not found")
        except Exception as e:
            print(f"Error importing {args.agent_name}: {e}")
            raise e

        test_agent(agent, env)        

if __name__ == "__main__":
    args = parse()
    run(args)