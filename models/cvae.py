# pytorch
import torch
from torch import nn

# python
from typing import Optional, Tuple

CVAE_DEFAULT_CONFIG = {
    # 6 if euler angles, 7 if quaternion
    "pose_dims": 6,
    # number of actuated joints 
    "joint_dims": 7,
    # dimension of hidden layer
    "hidden_dims": 50,
    # dimension of latent space
    "latent_dims": 2
}

class CVAE(nn.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        # updated config
        self._config = CVAE_DEFAULT_CONFIG
        if config is not None:
            self._config.update(config)
        # encoder takes desired pose + joint configuration
        self.encoder = nn.Sequential(
            nn.Linear(self._config["pose_dims"]+self._config["joint_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], 2*self._config["latent_dims"])
        )
        # decoder takes latent space + desired pose
        self.decoder = nn.Sequential(
            nn.Linear(self._config["latent_dims"]+self._config["pose_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["joint_dims"])
        )

    def forward(self, joint_config: torch.Tensor, desired_pose: torch.Tensor) -> Tuple[torch.Tensor]:
        # forward through encoder to get distribution params
        latent_params = self.encoder(torch.cat((joint_config, desired_pose)))
        # sample mean
        mean = latent_params[0:self._config["latent_dims"]]
        # convert log of variance to standard deviation
        log_variance = latent_params[self._config["latent_dims"]:]
        stddev = log_variance.mul(0.5).exp_()
        # noise sample from standard normal distribution
        std_norm_var = stddev.new_empty(stddev.size()).normal_()
        # reparameterization trick to sample from distribution
        z = std_norm_var.mul_(stddev).add_(mean)
        # run through decoder
        return self.decoder(torch.cat((z, desired_pose))), mean, log_variance

# for testing purposes
if __name__ == "__main__":
    device = torch.device("cpu")
    model = CVAE().to(device)
    model.train()
    pose = torch.Tensor([1,2,3,4,5,6])
    config = torch.Tensor([11,12,13,14,15,16,17])
    out, mean, log_variance = model(config, pose)