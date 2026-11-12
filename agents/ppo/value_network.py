import torch.nn as nn
from torch.nn.functional import tanh, relu
from torchrl.modules import ValueOperator

class PPOValueNetwork(nn.Module):
    """
    Initialize a value network for a PPO agent.
    """

    def __init__(self, in_channels=21):
        super(PPOValueNetwork, self).__init__()
        self.layer1 = nn.Linear(in_channels, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return x

def get_value_critic(network: nn.Module):
    return ValueOperator(network, in_keys=["state"])
