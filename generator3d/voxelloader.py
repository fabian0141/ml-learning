from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class VoxelDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, voxelSize):
        self.root = root
        self.voxelSize = voxelSize
        self.names = os.listdir(root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        voxels = np.load(self.root + "\\" + self.names[idx])
        size = len(voxels)
        voxels = np.transpose(voxels)
        values = np.full(shape=size, fill_value=255, dtype=np.uint8)
        voxels = torch.sparse_coo_tensor(voxels, values, (self.voxelSize, self.voxelSize, self.voxelSize))
        return voxels


if __name__ == '__main__':
    dataroot = "..\\DataSets\\MegaScans\\Voxel32"
    workers = 1
    batch_size = 8
    voxelSize = 32
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1
    kernelSize = 4


    dataset = VoxelDataSet(dataroot, voxelSize)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            print(data, data.shape, i)
            exit(0)