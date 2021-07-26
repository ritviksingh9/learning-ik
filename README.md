# Learning Inverse Kinematics

This implements a conditional VAE for the purpose of learning inverse kinematics for a 7 DOF Franka Panda robotic arm. Since a 7 DOF robotic arm has one degree of redundancy, there are generally multiple solutions for one given desired end-effector pose. However, traditional optimization based inverse kinematics algorithms, such as CLIK, can only yield one solution. 

The idea behind using a conditional VAE is that multiple inverse kinematics solutions may be acquired via traversing the latent space. With multiple solutions generated, one can then select the "best" one, where the definiton of "best" is dependent on the given situation (e.g. a solution that avoids collisions with obstacles or a solution that demands the least amount of actuation in the joint space). 

Note that if this model is to be re-trained for other robots, the URDF is required to perform forward kinematics for generating the dataset. 


## Setup Instructions
- Clone the repository
```
git clone --recurse-submodules https://github.com/ritviksingh9/learning-ik.git
```
- Follow the instructions in ```/differentiable-robot-model/README.md``` to install the library or type in the following command:
```
cd differentiable-robot-model
python setup.py develop
```
- Conda Dependencies:
  - [Pinocchio](https://stack-of-tasks.github.io/pinocchio/): Provides URDF parser and forward kinematics functions


## TO-DO
- Increase the beta value of the KL divergence term in the cost function to further regularize the latent space
- Extend this conditional VAE to other kinematically-redundant robots (e.g. KUKA IIWA 7)


