import torch.nn as nn
from torch.nn.functional import tanh, softmax

from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor

class PPOPolicyNetwork(nn.Module):
    """
    Initialize a policy network for a PPO agent.
    """

    def __init__(self, in_channels=21, num_actions=4):
        super(PPOPolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(in_channels, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = tanh(self.layer1(x))
        x = tanh(self.layer2(x))
        x = softmax(self.layer3(x), 0)
        return x

def get_policy_actor(network: nn.Module) -> ProbabilisticActor:
    policy_module = TensorDictModule(network, in_keys=["state"], out_keys=["probs"])
    policy_actor = ProbabilisticActor(
        module=policy_module, 
        in_keys=["probs"], 
        distribution_class=Categorical, 
        return_log_prob=True
        )
    return policy_actor