
"""
    This is a script that tests a U-Net
    Usage:
        python train.py --method 2D --net /path/to/network.pytorch
"""

"""
    Necessary libraries
"""
import numpy as np
import os
import argparse
import datetime

from data.epfl import EPFLDataset
from util.validation import segment
from util.tools import load_net
from util.io import imwrite3D, read_tif
from util.preprocessing import normalize

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--method", help="Specifies 2D or 3D U-Net", type=str, default="2D")

# logging parameters
parser.add_argument("--write_dir", help="Writing directory", type=str, default="output")

# network parameters
parser.add_argument("--data", help="Path to the data (should be tif file)", type=str, default="data/testing.tif")

# network parameters
parser.add_argument("--net", help="Path to the network", type=str, default="checkpoint.pytorch")

# optimization parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="512,512")
parser.add_argument("--batch_size", help="Batch size", type=int, default=1)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]

"""
    Setup writing directory
"""
print('[%s] Setting up write directories' % (datetime.datetime.now()))
if not os.path.exists(args.write_dir):
    os.mkdir(args.write_dir)

"""
    Load and normalize the data
"""
print('[%s] Loading and normalizing the data' % (datetime.datetime.now()))
test_data = read_tif(args.data, dtype='uint8')
mu = np.mean(test_data)
std = np.std(test_data)
test_data = normalize(test_data, mu, std)
if len(test_data.shape)<3:
    test_data = test_data[np.newaxis, ...]

"""
    Load the network
"""
print('[%s] Loading network' % (datetime.datetime.now()))
net = load_net(args.net)

"""
    Segmentation
"""
print('[%s] Starting segmentation' % (datetime.datetime.now()))
segmentation = segment(test_data, net, args.input_size, batch_size=args.batch_size)

"""
    Write out the results
"""
print('[%s] Writing the output' % (datetime.datetime.now()))
imwrite3D(segmentation, args.write_dir, rescale=True)

print('[%s] Finished!' % (datetime.datetime.now()))