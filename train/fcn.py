
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
import torch.optim as optim
from torch.utils.data import DataLoader

from data.epfl.epfl import EPFLDataset
from data.embl.embl import EMBLDataset
from networks.fcn import FCN2D8, FCN2D16, FCN2D32
from util.losses import cross_entropy2d
from util.preprocessing import get_augmenters_2d, get_augmenters_3d
from util.validation import segment
from util.metrics import jaccard, dice

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--data", help="Dataset for training", type=str, default="epfl") # options: 'epfl', 'embl_mito', 'embl_er'
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=10)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="512,512")
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=1)
parser.add_argument("--init", help="Use initialization of previous networks", type=int, default=1)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)
loss_fn_seg = cross_entropy2d

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.exists(os.path.join(args.log_dir,'FCN8')):
    os.mkdir(os.path.join(args.log_dir,'FCN8'))
if not os.path.exists(os.path.join(args.log_dir,'FCN16')):
    os.mkdir(os.path.join(args.log_dir,'FCN16'))
if not os.path.exists(os.path.join(args.log_dir,'FCN32')):
    os.mkdir(os.path.join(args.log_dir,'FCN32'))

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
# load data
print('[%s] Loading data' % (datetime.datetime.now()))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise==1))
if args.data == 'epfl':
    train = EPFLDataset(input_shape=input_shape, train=True,
                        transform=train_xtransform, target_transform=train_ytransform)
    test = EPFLDataset(input_shape=input_shape, train=False,
                       transform=test_xtransform, target_transform=test_ytransform)
else:
    mito = False
    if args.data == 'embl_mito':
        mito = True
    train = EMBLDataset(input_shape=input_shape, train=True,
                        transform=train_xtransform, target_transform=train_ytransform, mito=mito)
    test = EMBLDataset(input_shape=input_shape, train=False,
                       transform=test_xtransform, target_transform=test_ytransform, mito=mito)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.test_batch_size)


"""
    Setup optimization for training FCN8 network
"""
print('[%s] Setting up optimization for training FCN8 network' % (datetime.datetime.now()))
# load best checkpoint
net = FCN2D8()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the FCN8 network
"""
print('[%s] Training FCN8 network' % (datetime.datetime.now()))
net.train_net(train_loader=train_loader, test_loader=test_loader,
              loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=os.path.join(args.log_dir,'FCN8'))

"""
    Validating the trained FCN8 network
"""
print('[%s] Validating the trained FCN8 network' % (datetime.datetime.now()))
test_data = test.data
test_labels = test.labels
segmentation_last_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
print('[%s] FCN8 network performance (last checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
net = torch.load(os.path.join(args.log_dir,'FCN8','best_checkpoint.pytorch'))
segmentation_best_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
print('[%s] FCN8 network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))


"""
    Setup optimization for training FCN16 network
"""
print('[%s] Setting up optimization for training FCN16 network' % (datetime.datetime.now()))
if args.init:
    net = FCN2D16(fcn8s_weights=os.path.join(args.log_dir,'FCN8','best_checkpoint.pytorch'))
else:
    net = FCN2D16()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the FCN16 network
"""
print('[%s] Training FCN16 network' % (datetime.datetime.now()))
net.train_net(train_loader=train_loader, test_loader=test_loader,
              loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=os.path.join(args.log_dir,'FCN16'))

"""
    Validating the trained FCN16 network
"""
print('[%s] Validating the trained FCN16 network' % (datetime.datetime.now()))
test_data = test.data
test_labels = test.labels
segmentation_last_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
print('[%s] FCN16 network performance (last checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
net = torch.load(os.path.join(args.log_dir,'FCN16','best_checkpoint.pytorch'))
segmentation_best_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
print('[%s] FCN16 network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))


"""
    Setup optimization for training FCN32 network
"""
print('[%s] Setting up optimization for training FCN32 network' % (datetime.datetime.now()))
if args.init:
    net = FCN2D32(fcn8s_weights=os.path.join(args.log_dir,'FCN8','best_checkpoint.pytorch'))
else:
    net = FCN2D32()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the FCN32 network
"""
print('[%s] Training FCN32 network' % (datetime.datetime.now()))
net.train_net(train_loader=train_loader, test_loader=test_loader,
              loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=os.path.join(args.log_dir,'FCN32'))

"""
    Validating the trained FCN32 network
"""
print('[%s] Validating the trained FCN32 network' % (datetime.datetime.now()))
test_data = test.data
test_labels = test.labels
segmentation_last_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
print('[%s] FCN32 network performance (last checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
net = torch.load(os.path.join(args.log_dir,'FCN32','best_checkpoint.pytorch'))
segmentation_best_checkpoint = segment(test_data, net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
print('[%s] FCN32 network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))

print('[%s] Finished!' % (datetime.datetime.now()))