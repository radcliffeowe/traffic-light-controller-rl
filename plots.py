import argparse 
import json
import numpy as np
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description="Sumo RL Agents")
    parser.add_argument('--agent_name', required=True)
    parser.add_argument('--output_path', default=None)
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def plot_learning_curve(metrics, window=10, path="learning_curve.png"):
    def smooth_curve(x):
        x = np.asarray(x, dtype=float)
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    episode = np.asarray(metrics["episode"])
    ax = axes[0,0]
    ax.plot(episode, metrics["train_reward"], alpha=0.3, label="training rewards")
    ax.plot(episode[window-1:], smooth_curve(metrics["train_reward"]), label=f"training reward, {window} episodes average")
    ax.plot(metrics["eval_episode"], metrics["eval_reward"], "o-", label="evaluation reward")
    ax.set(title="Mean reward per step", xlabel="Episode")
    ax.legend()

    axes[0, 1].plot(episode, metrics["value_loss"])
    axes[0, 1].set(title="Value loss", xlabel="Episode")

    axes[1, 0].plot(episode, metrics["entropy"])
    axes[1, 0].set(title="Policy entropy", xlabel="Episode")

    axes[1, 1].plot(episode, metrics["clip_fraction"])
    axes[1, 1].set(title="Clip fraction", xlabel="Episode")

    fig.tight_layout()
    fig.savefig(path, dpi=150)

def load_metrics(agent_name):
    with open(f"agents/{agent_name}/results/training_results.json", "r") as f:
        metrics = json.load(f)
        return metrics

if __name__ == "__main__":
    args = parse()
    metrics = load_metrics(args.agent_name)
    plot_learning_curve(metrics, path=f"{args.agent_name}_learning_curve.png")
