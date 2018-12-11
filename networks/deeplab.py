
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

# original 2D DeepLab model
class DeepLab(nn.Module):

    def __init__(self, in_channels=1, out_channels=2):
        self.inplanes = 64
        super(DeepLab, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=1, dilation=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24], out_channels)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #         #        for i in m.parameters():
        #         #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        interp = nn.Upsample(size=loader.dataset.input_shape[1:], mode='bilinear', align_corners=True)

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = interp(self(x))

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

            if write_images:
                # write images
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/y', y, epoch)
                writer.add_image('train/y_pred', y_pred, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        interp = nn.Upsample(size=loader.dataset.input_shape[1:], mode='bilinear', align_corners=True)

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # forward prop
            y_pred = interp(self(x))

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

            if write_images:
                # write images
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/y', y, epoch)
                writer.add_image('test/y_pred', y_pred, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1):

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
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn, epoch=epoch, writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()