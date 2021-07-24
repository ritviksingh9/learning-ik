# pytorch
import torch
from torch import nn

# python
from typing import Optional, Tuple

CVAE_DEFAULT_CONFIG = {
    # 6 if euler angles, 7 if quaternion
    "pose_dims": 7,
    # number of actuated joints 
    "joint_dims": 7,
    # dimension of hidden layer
    "hidden_dims": 150,
    # dimension of latent space
    "latent_dims": 1,
    # lower joint position limits
    "lower_joint_limits": torch.Tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    # upper joint position limits
    "upper_joint_limits": torch.Tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]),
}


class ConstrainedCVAE(nn.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # updated config
        self._config = CVAE_DEFAULT_CONFIG
        if config is not None:
            self._config.update(config)
        # joint position range:
        self._joint_position_range = (self._config["upper_joint_limits"] - self._config["lower_joint_limits"]).to(self.device)
        # joint position mean:
        self._joint_position_mean = 0.5*(self._config["upper_joint_limits"] + self._config["lower_joint_limits"]).to(self.device)

        # encoder takes desired pose + joint configuration
        self.encoder = nn.Sequential(
            nn.Linear(self._config["pose_dims"] + self._config["joint_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
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
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["hidden_dims"]),
            nn.ReLU(),
            nn.Linear(self._config["hidden_dims"], self._config["joint_dims"]),
            nn.Tanh()
        )

    def forward(self, desired_pose: torch.Tensor, joint_config: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        if self.training:
            # forward through encoder to get distribution params
            latent_params = self.encoder(torch.cat((joint_config, desired_pose), axis=1).view(-1, self._config["pose_dims"] + self._config["joint_dims"]))
            # sample mean
            mean = latent_params[:, 0:self._config["latent_dims"]]
            # convert log of variance to standard deviation
            log_variance = latent_params[:, self._config["latent_dims"]:]
            stddev = log_variance.mul(0.5).exp_()
            # noise sample from standard normal distribution
            std_norm_var = stddev.new_empty(stddev.size()).normal_()
            # reparameterization trick to sample from distribution
            z = std_norm_var.mul_(stddev).add_(mean)
            # run through decoder
            output = self.decoder(torch.cat((z, desired_pose), axis=1))
            # scale the output to meet joint position constraints
            q = output*0.5*self._joint_position_range + self._joint_position_mean
            return q, mean, log_variance
        else:
            # if z is not provided, sample from standard normal
            if z == None:
                z = torch.empty(self._config["latent_dims"]).normal_(mean=0, std=1)
                # print("sampled z: ", z)
            z = z.to(self.device)
            # run through decoder
            output = self.decoder(torch.cat((z, desired_pose), axis=0))
            # scale the output to meet joint position constraints
            q = output*0.5*self._joint_position_range + self._joint_position_mean
            return q


# for testing purposes
if __name__ == "__main__":
    device = torch.device("cpu")
    model = CVAE().to(device)
    model.train()
    pose = torch.Tensor([1,2,3,4,5,6])
    config = torch.Tensor([11,12,13,14,15,16,17])
    out, mean, log_variance = model(config, pose)


