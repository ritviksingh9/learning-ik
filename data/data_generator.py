# pinocchio 
import pinocchio

# data generating config
from data import DataGenConfig

# python
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import dirname, abspath, join

def gen_rand_config(lower_limit: np.ndarray, upper_limit: np.ndarray) -> np.ndarray:
    return np.random.uniform(low=lower_limit, high=upper_limit)

def generate_data():
    # model paths
    pinocchio_model_dir = dirname(dirname(str(abspath(__file__)))) 
    model_path = pinocchio_model_dir + "/resources/"+DataGenConfig.ROBOT
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
    
    # setting up file writing
    file_name = DataGenConfig.OUT_FILE_NAME 
    file = open(file_name, "w")
    file.write("Pose\tConfiguration\n")

    num_data = DataGenConfig.NUM_DATA
    # data generating loop
    for i in range(num_data):
        # generating feature and label
        config = gen_rand_config(lower_limit, upper_limit)
        pinocchio.framesForwardKinematics(model, data, config)
        pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
        # converting quaternion to euler angle 
        if DataGenConfig.IS_QUAT:
            rotation = R.from_quat(list(pose[3:]))
            rotation_euler = rotation.as_euler("zxy")
            pose = np.concatenate((pose[0:3],rotation_euler))
        # annoying string manipulation for saving in text file
        str_pose = [str(i) for i in pose]
        str_config = [str(i) for i in config]
        file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")

    # close file buffer
    file.close()

if __name__ == "__main__":
    generate_data()
