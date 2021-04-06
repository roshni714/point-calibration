import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
import h5py
import pandas as pd

class SatelliteDataset(Dataset):

    def __init__(self):
        vb_dir = os.path.dirname(__file__)
        data_dir = os.path.join(vb_dir, "data/satellite")
        self.f = h5py.File('{}/satellite.h5'.format(data_dir), 'r')
        self.y_scale = pd.read_csv("{}/data.csv".format(data_dir))["label"].std() 

    def __getitem__(self, idx):
        return torch.Tensor(self.f[str(idx)]["image"].value)/255, torch.Tensor([self.f[str(idx)]["label"].value]) 

    def __len__(self):
        return len(self.f.keys())


def get_satellite_dataloaders(split_seed=0, batch_size=None, test_fraction=0.1, combine_val_train=False):
     
    dataset = SatelliteDataset()
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    rs = np.random.RandomState(split_seed)
    permutation = rs.permutation(dataset_size)

    size_train = int(np.round(dataset_size* (1 - test_fraction)))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    permutation = rs.permutation(len(index_train))
    if combine_val_train:
        val_fraction = 0.
    else:
        val_fraction = 0.15
    size_val = int(val_fraction * size_train)
    index_val = index_train[:size_val]
    index_train = index_train[size_val:]

    for x in index_train:
        assert x not in index_val
        assert x not in index_test

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(index_train)
    test_sampler = SubsetRandomSampler(index_test)
    val_sampler = SubsetRandomSampler(index_val)

    if not batch_size:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(index_train),
                                           sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=len(index_val),
                                                sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(index_test),
                                                sampler=test_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=len(index_val),
                                                sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(index_test),
                                                sampler=test_sampler)



    return train_loader, validation_loader, test_loader, np.array([dataset[0][0].shape[0]]), np.array([dataset[0][1].shape[0]]), dataset.y_scale  

if __name__ == "__main__":

    dataset = SatelliteDataset()
    min_num = float("inf")
    max_num = -float("inf")
    for i in range(len(dataset)):
        if torch.max(dataset[i][0]) > max_num:
            max_num = torch.max(dataset[i][0])
        if torch.min(dataset[i][0]) < min_num:
            min_num = torch.min(dataset[i][0])
    print(min_num, max_num)


