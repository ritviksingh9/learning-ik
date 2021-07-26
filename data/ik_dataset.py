# pytorch
import torch
from torch.utils.data import Dataset

# data generating config
from data import DataGenConfig
# from data_config import DataGenConfig

# python
import numpy as np

class IKDataset(Dataset):
    def __init__(self):
        dataset = np.loadtxt("data/" + DataGenConfig.OUT_FILE_NAME, 
                        delimiter=",",
                        dtype = np.float32,
                        skiprows=1)
        # dataset = np.loadtxt(DataGenConfig.OUT_FILE_NAME, 
        #                 delimiter=",",
        #                 dtype = np.float32,
        #                 skiprows=1)

        if DataGenConfig.IS_QUAT:
            pose = np.array([data[0:7] for data in dataset])
            configuration = np.array([data[7:] for data in dataset])
        else:
            pose = np.array([data[0:6] for data in dataset])
            configuration = np.array([data[6:] for data in dataset])

        # converting to torch tensors
        self.x = torch.from_numpy(pose)
        self.y = torch.from_numpy(configuration)
        self.n_samples = dataset.shape[0]
    
    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

# for testing purposes
if __name__ == "__main__":
    dataset = IKDataset()
    pose, configuration = dataset[0]
    print(pose, configuration)
