# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import CVAE
from data import IKDataset 

# training hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 3e-6

def cvae_loss(joint_config: torch.Tensor, true_joint_config: torch.Tensor, 
                mean: torch.Tensor, log_variance: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(joint_config, true_joint_config, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss

def train():
    # fix this
    device = torch.device("cpu")
    cvae = CVAE().to(device)
    cvae.train()
    # generate dataset
    dataset = IKDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    # optimizer
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE,
                                    gamma=LEARNING_RATE_DECAY)
    # training loop
    for epoch in range(NUM_EPOCHS):
        epoch_error = 0
        for pose, joint_config in train_loader:
            joint_config_pred, mean, log_variance = cvae(joint_config, pose)
            loss = cvae_loss(joint_config_pred, joint_config,
                                mean, log_variance, 0.02)
            epoch_error += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()