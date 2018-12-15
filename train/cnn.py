
"""
    This is a script that trains U-Nets from scratch
    Usage:
        python train.py --method 2D
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.epfl import EPFLPixelDataset
from networks.cnn import CNN
from util.preprocessing import get_augmenters_2d
from util.validation import segment_pixels
from util.metrics import jaccard, dice, accuracy_metrics

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=1)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="95,95")
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=1)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=400)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=256)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=128)
loss_fn_seg = nn.CrossEntropyLoss()

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
# load data
print('[%s] Loading data (EPFL)' % (datetime.datetime.now()))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise==1))
train = EPFLPixelDataset(input_shape=input_shape, train=True,
                         transform=train_xtransform, target_transform=train_ytransform)
test = EPFLPixelDataset(input_shape=input_shape, train=False,
                        transform=test_xtransform, target_transform=test_ytransform)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.test_batch_size)

"""
    Setup optimization for finetuning
"""
print('[%s] Setting up optimization for finetuning' % (datetime.datetime.now()))
# load best checkpoint
net = CNN()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network
"""
print('[%s] Training network' % (datetime.datetime.now()))
net.train_net(train_loader=train_loader, test_loader=test_loader,
              loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=args.log_dir)

"""
    Validate the trained network
"""
print('[%s] Validating the trained network' % (datetime.datetime.now()))
test_data = test.data
test_labels = test.labels
segmentation_last_checkpoint = segment_pixels(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
a, p, r, f = accuracy_metrics(segmentation_last_checkpoint, test_labels)
print('[%s] RESULTS:' % (datetime.datetime.now()))
print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
print('[%s]     F-score: %f' % (datetime.datetime.now(), f))
net = torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
segmentation_best_checkpoint = segment_pixels(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
a, p, r, f = accuracy_metrics(segmentation_best_checkpoint, test_labels)
print('[%s] RESULTS:' % (datetime.datetime.now()))
print('[%s]     Jaccard: %f' % (datetime.datetime.now(), j))
print('[%s]     Dice: %f' % (datetime.datetime.now(), d))
print('[%s]     Accuracy: %f' % (datetime.datetime.now(), a))
print('[%s]     Precision: %f' % (datetime.datetime.now(), p))
print('[%s]     Recall: %f' % (datetime.datetime.now(), r))
print('[%s]     F-score: %f' % (datetime.datetime.now(), f))
print('[%s] Network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))

print('[%s] Finished!' % (datetime.datetime.now()))