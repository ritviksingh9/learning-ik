# pytorch
import torch
from torch.utils.data import DataLoader

# differentiable-robot-model
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableKUKAiiwa,
)

# python
import numpy as np
import os
import random

def test_forward_kinematics():
    urdf_path = "resources/franka/urdf/panda_arm.urdf"
    robot_model = DifferentiableRobotModel(
        urdf_path, name="franka_panda", device="cpu"
    )
    print(robot_model)
    q = torch.tensor([[1.,1.,1.,1.,1.,1.,1.]], device="cpu", requires_grad=True)
    link_name = "panda_link7"
    print(robot_model.compute_forward_kinematics(q, link_name))

if __name__ == "__main__":
    test_forward_kinematics()