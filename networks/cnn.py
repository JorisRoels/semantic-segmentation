
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
from tensorboardX import SummaryWriter

# original 2D CNN model
class CNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=2):
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # conv1
        self.conv1 = nn.Conv2d(in_channels, 48, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv2
        self.conv2 = nn.Conv2d(48, 48, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv3
        self.conv3 = nn.Conv2d(48, 48, 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4 = nn.Conv2d(48, 48, 4)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc5
        self.fc5 = nn.Linear(3*3*48, 200)
        self.relu5 = nn.ReLU(inplace=True)

        # fc6
        self.fc6 = nn.Linear(200, out_channels)

    def forward(self, x):

        h = x

        h = self.relu1(self.conv1(h))
        h = self.pool1(h)

        h = self.relu2(self.conv2(h))
        h = self.pool2(h)

        h = self.relu3(self.conv3(h))
        h = self.pool3(h)

        h = self.relu4(self.conv4(h))
        h = self.pool4(h)

        h = h.view(h.size(0),-1)

        h = self.relu5(self.fc5(h))

        h = self.fc6(h)

        return h

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader.dataset), loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average train loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('train/loss', loss_avg, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss', loss_avg, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                             print_stats=print_stats, writer=writer)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()