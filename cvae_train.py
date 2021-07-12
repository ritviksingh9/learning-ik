# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import CVAE
from data import IKDataset 

# training hyperparameters
BATCH_SIZE = 100
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 3e-6

def cvae_loss(joint_config: torch.Tensor, true_joint_config: torch.Tensor, 
              mean: torch.Tensor, log_variance: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(joint_config, true_joint_config, reduction="sum")
    kl_loss = 0.5 * torch.sum(log_variance.exp() + mean.pow(2) - 1. - log_variance)
    return recon_loss + beta*kl_loss

def train():
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = CVAE().to(device)
    cvae.train()
    # generate dataset
    dataset = IKDataset()
    # shuffle=False because each data point is already randomly sampled
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    # optimizer
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)
    # training loop
    for epoch in range(NUM_EPOCHS):
        epoch_error = 0
        for pose, joint_config in train_loader:
            pose = pose.to(device)
            joint_config = joint_config.to(device)
            joint_config_pred, mean, log_variance = cvae(joint_config, pose)
            loss = cvae_loss(joint_config_pred, joint_config,
                                mean, log_variance, 0.02)
            epoch_error += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        optimizer_scheduler.step()
        print("Epoch Number: {} || Average Error: {}".format(epoch, epoch_error/dataset.n_samples))


if __name__ == "__main__":
    train()