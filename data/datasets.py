
import os
import numpy as np
import torch.utils.data as data
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input

class VolumeDataset(data.Dataset):

    def __init__(self, data_path, label_path, input_shape, len_epoch=1000, transform=None, target_transform=None):

        self.data_path = data_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data = read_tif(data_path, dtype='uint8')
        self.labels = read_tif(label_path, dtype='uint8')

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

class EPFLTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None):
        super(EPFLTrainDataset, self).__init__(os.path.join('../data', 'epfl', 'training.tif'),
                                               os.path.join('../data', 'epfl', 'training_groundtruth.tif'),
                                               input_shape,
                                               len_epoch=len_epoch,
                                               transform=transform,
                                               target_transform=target_transform)

class EPFLTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None):
        super(EPFLTestDataset, self).__init__(os.path.join('../data', 'epfl', 'testing.tif'),
                                              os.path.join('../data', 'epfl', 'testing_groundtruth.tif'),
                                              input_shape,
                                              len_epoch=len_epoch,
                                              transform=transform,
                                              target_transform=target_transform)

class EPFLPixelTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None):
        super(EPFLPixelTrainDataset, self).__init__(os.path.join('../data', 'epfl', 'training.tif'),
                                               os.path.join('../data', 'epfl', 'training_groundtruth.tif'),
                                               input_shape,
                                               len_epoch=len_epoch,
                                               transform=transform,
                                               target_transform=target_transform)

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

class EPFLPixelTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None):
        super(EPFLPixelTestDataset, self).__init__(os.path.join('../data', 'epfl', 'testing.tif'),
                                                    os.path.join('../data', 'epfl', 'testing_groundtruth.tif'),
                                                    input_shape,
                                                    len_epoch=len_epoch,
                                                    transform=transform,
                                                    target_transform=target_transform)

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

class EMBLMitoTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLMitoTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[:s, :, :]

class EMBLMitoTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLMitoTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[s:, :, :]

class EMBLERTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLERTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'er_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[:s, :, :]

class EMBLERTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLERTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                  os.path.join('../data', 'embl', 'er_labels.tif'),
                                                  input_shape,
                                                  len_epoch=len_epoch,
                                                  transform=transform,
                                                  target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[s:, :, :]

class EMBLMitoPixelTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLMitoPixelTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[:s, :, :]

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

class EMBLMitoPixelTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLMitoPixelTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[s:, :, :]

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

class EMBLERPixelTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLERPixelTrainDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                   os.path.join('../data', 'embl', 'er_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[:s, :, :]

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

class EMBLERPixelTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(EMBLERPixelTestDataset, self).__init__(os.path.join('../data', 'embl', 'data.tif'),
                                                  os.path.join('../data', 'embl', 'er_labels.tif'),
                                                  input_shape,
                                                  len_epoch=len_epoch,
                                                  transform=transform,
                                                  target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[s:, :, :]

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

class VNCTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(VNCTrainDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[:s, :, :]

class VNCTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(VNCTestDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[s:, :, :]

class VNCPixelTrainDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(VNCPixelTrainDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[:s, :, :]

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

class VNCPixelTestDataset(VolumeDataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None, target_transform=None, split=0.5):
        super(VNCPixelTestDataset, self).__init__(os.path.join('../data', 'vnc', 'data.tif'),
                                                   os.path.join('../data', 'vnc', 'mito_labels.tif'),
                                                   input_shape,
                                                   len_epoch=len_epoch,
                                                   transform=transform,
                                                   target_transform=target_transform)

        s = int(split * self.data.shape[0])
        self.data = self.data[s:, :, :]

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