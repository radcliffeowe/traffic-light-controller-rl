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
    parser.add_argument('--use_gui', action='store_true')
    parser.add_argument('--num_seconds', default=1000)
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args):
    env = sumo_rl.SumoEnvironment(
            net_file=args.net_file,
            route_file=args.route_file,
            out_csv_name=args.output_path,
            single_agent=True,
            use_gui=args.use_gui,
            num_seconds=args.num_seconds,
            reward_fn="queue"
            )
    if args.test_agent:
        try:
            if args.agent_name == "random":
                from agents.random_agent import RandomAgent
                agent = RandomAgent(env)
            elif args.agent_name == "greedy":
                from agents.greedy_agent import GreedyAgent
                agent = GreedyAgent(env)
            elif args.agent_name == "ppo":
                from agents.ppo.ppo_agent import PPOAgent
                agent = PPOAgent(env)
            else:
                print("Agent not found, exiting...")
                raise Exception("Agent not found")
        except Exception as e:
            raise e

        test_agent(agent, env, output_path=args.output_path) 
    else:
        try:
            if args.agent_name == "ppo":
                from agents.ppo.ppo_agent import PPOAgent
                agent = PPOAgent(env)
            else:
                raise Exception(f"Cannot train {args.agent_name} agent")
            
            agent.train()
        except Exception as e:
            raise e

if __name__ == "__main__":
    args = parse()
    run(args)