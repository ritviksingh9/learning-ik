import dataclasses

@dataclasses.dataclass
class DataGenConfig:
    """
    @brief Defines basic configuration parameters for generating the dataset.
    """
    # Basic robot model information 
    ROBOT: str = "franka"
    ROBOT_URDF: str = "panda_arm.urdf"
    EE_NAME: str = "panda_link7"
    JOINT_DIMS: int = 7
    # Config for the dataset
#    OUT_FILE_NAME: str = "franka_ik_data_5_dof.txt" 
    OUT_FILE_NAME: str = "franka_ik_data.txt"
    IS_QUAT: bool = True
    NUM_DATA: int = 100000
