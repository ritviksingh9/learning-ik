# pytorch
import pinocchio
from data.data_config import DataGenConfig

import torch
from torch import nn
import numpy as np
from os.path import dirname, abspath, join

from model import ConstrainedCVAE




def inference(pose, z = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae = ConstrainedCVAE().to(device)
    cvae.load_state_dict(torch.load("model/weights/constrained_cvae_weights.pth", map_location=device))

    cvae.eval()

    with torch.no_grad():
        pose = pose.to(device)
        q = cvae(desired_pose = pose, z = z)
        return q



if __name__ == "__main__":
    # pose = torch.Tensor([-0.3622564536646905,0.07453657615711093,0.523455111826844,0.6949510146762841,0.6371909076037253,-0.28704989069534576,-0.16921345903719948])
    pose = torch.Tensor([0.5, 0, 0.5, 0, 0, 0, 1])
    


    z = None
    #z = torch.Tensor([0, 0, 0])
    q = inference(pose=pose, z=z)

    print("Generated q: ", q)


    pinocchio_model_dir = dirname(dirname(str(abspath(__file__)))) 
    model_path = pinocchio_model_dir + "/learning-ik/resources/" + DataGenConfig.ROBOT
    urdf_path = model_path + "/urdf/"+DataGenConfig.ROBOT_URDF
    # setup robot model and data
    model = pinocchio.buildModelFromUrdf(urdf_path)
    data = model.createData()
    # setup end effector
    ee_name = DataGenConfig.EE_NAME
    ee_link_id = model.getFrameId(ee_name)
    # joint limits (from urdf)
    lower_limit = np.array(model.lowerPositionLimit)
    upper_limit = np.array(model.upperPositionLimit)
    pinocchio.framesForwardKinematics(model, data, q.cpu().numpy())
    desired_pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])

    print("Desired Pose", pose[:3].cpu().numpy())
    print("Generated Pose: ", desired_pose[:3])
    print("Error: ", np.linalg.norm(pose[:3].cpu().numpy() - desired_pose[:3]))






    print("------------------------------------------------------\n")

    for i in range (10):
        z = torch.Tensor([2*i-1])
        q = inference(pose=pose, z=z)
        pinocchio.framesForwardKinematics(model, data, q.cpu().numpy())
        desired_pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
        print(list(q.cpu().numpy()))
        print("Desired Pose", pose[:3].cpu().numpy())
        print("Generated Pose: ", desired_pose[:3])
        print("Error: ", np.linalg.norm(pose[:3].cpu().numpy() - desired_pose[:3]))
        print("\n")

