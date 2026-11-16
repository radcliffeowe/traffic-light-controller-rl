from tqdm import tqdm
import os
import numpy as np

from metrics import MetricsCollector

def test_agent(agent, env, num_episodes=5, seed=42, output_path=None):
    print("Testing agent...")
    metrics_collector = MetricsCollector(env)
    total_rewards = []
    for episode in tqdm(range(num_episodes)):
        truncated=False
        terminated=False
        episode_rewards = []
        state = env.reset(seed)
        metrics_collector.env = env
        while not terminated and not truncated:
            if isinstance(state, tuple):
                state = state[0]  # first observation sometimes includes info dict
            action = agent.make_action(state, test=True)
            if isinstance(action, tuple):
                action = action[0]
            state, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            metrics_collector.record_metrics()
        total_rewards.append(np.mean(episode_rewards))
    df = metrics_collector.get_summary()
    if output_path:
        df = metrics_collector.get_summary()
        df.to_csv(f"{os.path.dirname(output_path)}/{agent.__class__.__name__}_episode_{episode}.csv")
    return np.nanmean(total_rewards), total_rewards


