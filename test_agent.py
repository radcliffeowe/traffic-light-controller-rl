from tqdm import tqdm


def test_agent(agent, env, num_episodes=5, seed=42):
    print("Testing agent...")
    for episode in tqdm(range(num_episodes)):
        truncated=False
        terminated=False
        state = env.reset()
        while not terminated and not truncated:
            action = agent.make_action(state, test=True)
            state, reward, terminated, truncated, info = env.step(action)
    env.close()

