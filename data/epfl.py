
import os
import numpy as np
import random
import torch.utils.data as data
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input

class EPFLDataset(data.Dataset):

    def __init__(self, input_shape, train=True,
                 len_epoch=1000, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data = read_tif(os.path.join('../data', 'epfl', 'training.tif'), dtype='uint8')
            self.labels = read_tif(os.path.join('../data', 'epfl', 'training_groundtruth.tif'), dtype='int')
        else:
            self.data = read_tif(os.path.join('../data', 'epfl', 'testing.tif'), dtype='uint8')
            self.labels = read_tif(os.path.join('../data', 'epfl', 'testing_groundtruth.tif'), dtype='int')

        # normalize data
        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.data = normalize(self.data, mu, std)
        self.labels = normalize(self.labels, 0, 255)

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target[np.newaxis, ...]
        else:
            return input, target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

class EPFLPixelDataset(data.Dataset):

    def __init__(self, input_shape, train=True,
                 len_epoch=1000, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data = read_tif(os.path.join('../data', 'epfl', 'training.tif'), dtype='uint8')
            self.labels = read_tif(os.path.join('../data', 'epfl', 'training_groundtruth.tif'), dtype='int')
        else:
            self.data = read_tif(os.path.join('../data', 'epfl', 'testing.tif'), dtype='uint8')
            self.labels = read_tif(os.path.join('../data', 'epfl', 'testing_groundtruth.tif'), dtype='int')

        # normalize data
        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.data = normalize(self.data, mu, std)
        self.labels = normalize(self.labels, 0, 255)

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        target = target[target.shape[0]//2, target.shape[1]//2, target.shape[2]//2]
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target
        else:
            return input, target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std