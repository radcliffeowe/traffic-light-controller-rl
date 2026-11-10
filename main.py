import argparse

from test_agent import test_agent

import gymnasium as gym
import sumo_rl

def parse():
    parser = argparse.ArgumentParser(description="Sumo RL Agents")
    parser.add_argument('--agent_name', required=True)
    parser.add_argument('--test_agent', action='store_true')
    parser.add_argument('--net_file')
    parser.add_argument('--route_file')
    parser.add_argument('--output_path', default=None)
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args):
    if args.test_agent:
        env = gym.make('sumo-rl-v0',
                net_file=args.net_file,
                route_file=args.route_file,
                out_csv_name=args.output_path,
                single_agent=True,
                use_gui=False,
                num_seconds=100)

        try:
            if args.agent_name == "random":
                from agents.random_agent import RandomAgent
                agent = RandomAgent(env)
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