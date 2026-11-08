from tqdm import tqdm
import os

from metrics import MetricsCollector

def test_agent(agent, env, num_episodes=5, seed=42, output_path=None):
    print("Testing agent...")

    for episode in tqdm(range(num_episodes)):
        truncated=False
        terminated=False
        state = env.reset()
        metrics_collector = MetricsCollector(env)
        while not terminated and not truncated:
            if isinstance(state, tuple):
                state = state[0]  # first observation sometimes includes info dict
            action = agent.make_action(state, test=True)
            state, reward, terminated, truncated, info = env.step(action)
            metrics_collector.record_metrics()
        df = metrics_collector.get_summary()
        if output_path:
            df = metrics_collector.get_summary()
            df.to_csv(f"{os.path.dirname(output_path)}/{agent.__class__.__name__}_episode_{episode}.csv")
    env.close()


