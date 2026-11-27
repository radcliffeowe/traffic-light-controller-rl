import argparse

from test_agent import test_agent
from rewards import get_mean_queued

import gymnasium as gym
import sumo_rl

def parse():
    parser = argparse.ArgumentParser(description="Sumo RL Agents")
    parser.add_argument('--agent_name', required=True)
    parser.add_argument('--test_agent', action='store_true')
    parser.add_argument('--net_file', default='networks/two-way-single-intersection/single-intersection.net.xml')
    # parser.add_argument('--route_file', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--use_gui', action='store_true')
    parser.add_argument('--num_seconds', default=3600)
    parser.add_argument('--num_episodes', type=int, default=300)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run(args):
    if args.agent_name == "reinforce_baseline":
        import torch
        if args.test_agent and not args.checkpoint:
            raise ValueError("--checkpoint is required to evaluate reinforce_baseline")
        if min(args.num_episodes, args.eval_interval, int(args.num_seconds)) < 1:
            raise ValueError("Episode count, evaluation interval, and duration must be positive")
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)

    standard_env = sumo_rl.SumoEnvironment(
            net_file=args.net_file,
            route_file="networks/two-way-single-intersection/scenario-normal.rou.xml",
            single_agent=True,
            use_gui=args.use_gui,
            num_seconds=int(args.num_seconds),
            reward_fn=get_mean_queued
            )
    if args.test_agent:
        try:
            if args.agent_name == "fixed":
                from agents.fixed_time_agent import FixedTimeAgent
                agent = FixedTimeAgent(standard_env)
                print("initializing fixed agent")
            elif args.agent_name == "random":
                from agents.random_agent import RandomAgent
                agent = RandomAgent(standard_env)
            elif args.agent_name == "greedy":
                from agents.greedy_agent import GreedyAgent
                agent = GreedyAgent(standard_env)
            elif args.agent_name == "reinforce_baseline":
                from agents.reinforce_baseline_agent import ReinforceBaselineAgent
                agent = ReinforceBaselineAgent(standard_env)
                agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
                agent.eval()
            elif args.agent_name == "ppo":
                from agents.ppo.ppo_agent import PPOAgent
                agent = PPOAgent(standard_env)
                agent.load_agent()
            else:
                print("Agent not found, exiting...")
                raise Exception("Agent not found")
        except Exception as e:
            raise e

        mean_reward, _ = test_agent(agent, standard_env, seed=args.seed, output_path=args.output_path)
        print()
        print(f"EVAL {args.agent_name} mean reward: {mean_reward}")
        standard_env.close()
    else:
        heavy_env = sumo_rl.SumoEnvironment(
                net_file=args.net_file,
                route_file="networks/two-way-single-intersection/scenario-heavy.rou.xml",
                single_agent=True,
                use_gui=args.use_gui,
                num_seconds=int(args.num_seconds),
                reward_fn=get_mean_queued
                )
        imbalanced_env = sumo_rl.SumoEnvironment(
                net_file=args.net_file,
                route_file="networks/two-way-single-intersection/scenario-imbalanced.rou.xml",
                single_agent=True,
                use_gui=args.use_gui,
                num_seconds=int(args.num_seconds),
                reward_fn=get_mean_queued
                )
        varying_env = sumo_rl.SumoEnvironment(
                net_file=args.net_file,
                route_file="networks/two-way-single-intersection/scenario-time-varying.rou.xml",
                single_agent=True,
                use_gui=args.use_gui,
                num_seconds=int(args.num_seconds),
                reward_fn=get_mean_queued
                )
        envs = [standard_env, heavy_env, imbalanced_env, varying_env]
        try:
            if args.agent_name == "ppo":
                from agents.ppo.ppo_agent import PPOAgent
                agent = PPOAgent(envs)
            elif args.agent_name == "reinforce_baseline":
                from agents.reinforce_baseline_agent import ReinforceBaselineAgent
                from train_reinforce import train_agent
                agent = ReinforceBaselineAgent(envs)
                try:
                    train_agent(
                        agent, envs, args.output_path or "outputs/reinforce-baseline",
                        args.num_episodes, args.eval_interval, args.seed,
                    )
                finally:
                    for env in envs:
                        env.close()
                return
            else:
                raise Exception(f"Cannot train {args.agent_name} agent")

            agent.train()
            for env in envs:
                env.close()
        except Exception as e:
            raise e

if __name__ == "__main__":
    args = parse()
    run(args)