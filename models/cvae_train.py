# pytorch
import torch
from torch import nn

from cvae import CVAE
# fix this error!
from data.ik_dataset import IKDataset 

def cvae_loss(joint_config: torch.Tensor, true_joint_config: torch.Tensor, 
                mean: torch.Tensor, log_variance: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(joint_config, true_joint_config, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss

def train():
    # fix this
    device = torch.device("cpu")
    model = CVAE().to(device)
    model.train()

if __name__ == "__main__":
    dataset = IKDataset()
    pose, configuration = dataset[0]
    print(pose, configuration)