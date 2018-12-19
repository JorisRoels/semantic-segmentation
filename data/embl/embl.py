
import os
import numpy as np
import random
import torch.utils.data as data
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input

class EMBLDataset(data.Dataset):

    def __init__(self, input_shape, train=True,
                 len_epoch=1000, transform=None, target_transform=None, split=0.5, mito=False):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data = read_tif(os.path.join('../../data', 'embl', 'data.tif'), dtype='uint8')
        if mito:
            self.labels = read_tif(os.path.join('../../data', 'embl', 'mito_labels.tif'), dtype='int')
        else:
            self.labels = read_tif(os.path.join('../../data', 'embl', 'er_labels.tif'), dtype='int')
        s = int(split * self.data.shape[0])
        if self.train:
            self.data = self.data[:s, :, :]
            self.labels = self.labels[:s, :, :]
        else:
            self.data = self.data[s:, :, :]
            self.labels = self.labels[s:, :, :]

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

class EMBLPixelDataset(data.Dataset):

    def __init__(self, input_shape, train=True,
                 len_epoch=1000, transform=None, target_transform=None, split=0.5, mito=False):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data = read_tif(os.path.join('../../data', 'embl', 'data.tif'), dtype='uint8')
        if mito:
            self.labels = read_tif(os.path.join('../../data', 'embl', 'mito_labels.tif'), dtype='int')
        else:
            self.labels = read_tif(os.path.join('../../data', 'embl', 'er_labels.tif'), dtype='int')
        s = int(split * self.data.shape[0])
        if self.train:
            self.data = self.data[:s, :, :]
            self.labels = self.labels[:s, :, :]
        else:
            self.data = self.data[s:, :, :]
            self.labels = self.labels[s:, :, :]

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