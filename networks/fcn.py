
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from util.metrics import jaccard, accuracy_metrics

UNSUPERVISED_TRAINING = 0
SUPERVISED_TRAINING = 1

# original 2D FCN8 model
class FCN2D8(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, pretrain_unsupervised=False):
        super(FCN2D8, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrain_unsupervised = pretrain_unsupervised
        self.phase = SUPERVISED_TRAINING

        # conv1
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, out_channels, 1)
        self.score_pool3 = nn.Conv2d(256, out_channels, 1)
        self.score_pool4 = nn.Conv2d(512, out_channels, 1)

        self.upscore2 = nn.ConvTranspose2d(
            out_channels, out_channels, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            out_channels, out_channels, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            out_channels, out_channels, 4, stride=2, bias=False)

        if pretrain_unsupervised:
            self.phase = UNSUPERVISED_TRAINING

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        if self.phase == SUPERVISED_TRAINING:
            return h
        else:
            return h[:,0:self.in_channels,:,:]

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            if self.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
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
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('train/loss-seg', loss_avg, epoch)
            else:
                writer.add_scalar('train/loss-rec', loss_avg, epoch)

            if write_images:
                # write images
                if self.phase == SUPERVISED_TRAINING:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('train/x', x, epoch)
                    writer.add_image('train/y', y, epoch)
                    writer.add_image('train/y_pred', y_pred, epoch)
                else:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('train/x-rec-input', x, epoch)
                    writer.add_image('train/x-rec-output', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        j_cum = 0.0
        a_cum = 0.0
        p_cum = 0.0
        r_cum = 0.0
        f_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            if self.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            if self.phase == SUPERVISED_TRAINING:
                # compute other interesting metrics
                y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:,1,...]
                j_cum += jaccard(y_, y.cpu().numpy())
                a, p, r, f = accuracy_metrics(y_, y.cpu().numpy())
                a_cum += a; p_cum += p; r_cum += r; f_cum += f

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('test/loss-seg', loss_avg, epoch)
                writer.add_scalar('test/jaccard', j_avg, epoch)
                writer.add_scalar('test/accuracy', a_avg, epoch)
                writer.add_scalar('test/precision', p_avg, epoch)
                writer.add_scalar('test/recall', r_avg, epoch)
                writer.add_scalar('test/f-score', f_avg, epoch)
                if write_images:
                    # write images
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('test/x', x, epoch)
                    writer.add_image('test/y', y, epoch)
                    writer.add_image('test/y_pred', y_pred, epoch)
            else:
                writer.add_scalar('test/loss-rec', loss_avg, epoch)
                if write_images:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('test/x-rec-input', x, epoch)
                    writer.add_image('test/x-rec-output', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn_seg, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1,
                  train_loader_unsupervised=None, test_loader_unsupervised=None, loss_fn_rec=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        if self.pretrain_unsupervised:

            print('[%s] Starting unsupervised pre-training' % (datetime.datetime.now()))

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader_unsupervised, loss_fn=loss_fn_rec, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader_unsupervised, loss_fn=loss_fn_rec, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint_rec.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch_rec'))

            print('[%s] Starting supervised pre-training' % (datetime.datetime.now()))
            self.phase = SUPERVISED_TRAINING

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader, loss_fn=loss_fn_seg, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn_seg, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

            writer.close()

# original 2D FCN16 model
class FCN2D16(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, fcn8s_weights=None, pretrain_unsupervised=False):
        super(FCN2D16, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrain_unsupervised = pretrain_unsupervised
        self.phase = SUPERVISED_TRAINING

        # conv1
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, out_channels, 1)
        self.score_pool4 = nn.Conv2d(512, out_channels, 1)

        self.upscore2 = nn.ConvTranspose2d(
            out_channels, out_channels, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            out_channels, out_channels, 32, stride=16, bias=False)

        if fcn8s_weights is not None:
            fcn8s = torch.load(fcn8s_weights)
            self.copy_params_from_fcn8s(fcn8s)

        if pretrain_unsupervised:
            self.phase = UNSUPERVISED_TRAINING

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c

        h = self.upscore16(h)
        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()

        if self.phase == SUPERVISED_TRAINING:
            return h
        else:
            return h[:,0:self.in_channels,:,:]

    def copy_params_from_fcn8s(self, fcn8s):
        for name, l1 in fcn8s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            if self.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
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
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('train/loss-seg', loss_avg, epoch)
            else:
                writer.add_scalar('train/loss-rec', loss_avg, epoch)

            if write_images:
                # write images
                if self.phase == SUPERVISED_TRAINING:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('train/x', x, epoch)
                    writer.add_image('train/y', y, epoch)
                    writer.add_image('train/y_pred', y_pred, epoch)
                else:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('train/x-rec-input', x, epoch)
                    writer.add_image('train/x-rec-output', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        j_cum = 0.0
        a_cum = 0.0
        p_cum = 0.0
        r_cum = 0.0
        f_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            if self.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            if self.phase == SUPERVISED_TRAINING:
                # compute other interesting metrics
                y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:,1,...]
                j_cum += jaccard(y_, y.cpu().numpy())
                a, p, r, f = accuracy_metrics(y_, y.cpu().numpy())
                a_cum += a; p_cum += p; r_cum += r; f_cum += f

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('test/loss-seg', loss_avg, epoch)
                writer.add_scalar('test/jaccard', j_avg, epoch)
                writer.add_scalar('test/accuracy', a_avg, epoch)
                writer.add_scalar('test/precision', p_avg, epoch)
                writer.add_scalar('test/recall', r_avg, epoch)
                writer.add_scalar('test/f-score', f_avg, epoch)
                if write_images:
                    # write images
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('test/x', x, epoch)
                    writer.add_image('test/y', y, epoch)
                    writer.add_image('test/y_pred', y_pred, epoch)
            else:
                writer.add_scalar('test/loss-rec', loss_avg, epoch)
                if write_images:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('test/x-rec-input', x, epoch)
                    writer.add_image('test/x-rec-output', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn_seg, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1,
                  train_loader_unsupervised=None, test_loader_unsupervised=None, loss_fn_rec=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        if self.pretrain_unsupervised:

            print('[%s] Starting unsupervised pre-training' % (datetime.datetime.now()))

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader_unsupervised, loss_fn=loss_fn_rec, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader_unsupervised, loss_fn=loss_fn_rec, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint_rec.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch_rec'))

            print('[%s] Starting supervised pre-training' % (datetime.datetime.now()))
            self.phase = SUPERVISED_TRAINING

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader, loss_fn=loss_fn_seg, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn_seg, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

            writer.close()

# original 2D FCN32 model
class FCN2D32(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, fcn8s_weights=None, pretrain_unsupervised=False):
        super(FCN2D32, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrain_unsupervised = pretrain_unsupervised
        self.phase = SUPERVISED_TRAINING

        # conv1
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, out_channels, 1)
        self.upscore = nn.ConvTranspose2d(out_channels, out_channels, 64, stride=32,
                                          bias=False)

        if fcn8s_weights is not None:
            fcn8s = torch.load(fcn8s_weights)
            self.copy_params_from_fcn8s(fcn8s)

        if pretrain_unsupervised:
            self.phase = UNSUPERVISED_TRAINING

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        if self.phase == SUPERVISED_TRAINING:
            return h
        else:
            return h[:,0:self.in_channels,:,:]

    def copy_params_from_fcn8s(self, fcn8s):
        for name, l1 in fcn8s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            if self.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
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
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('train/loss-seg', loss_avg, epoch)
            else:
                writer.add_scalar('train/loss-rec', loss_avg, epoch)

            if write_images:
                # write images
                if self.phase == SUPERVISED_TRAINING:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('train/x', x, epoch)
                    writer.add_image('train/y', y, epoch)
                    writer.add_image('train/y_pred', y_pred, epoch)
                else:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('train/x-rec-input', x, epoch)
                    writer.add_image('train/x-rec-output', x_rec, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss and metrics during the epoch
        loss_cum = 0.0
        j_cum = 0.0
        a_cum = 0.0
        p_cum = 0.0
        r_cum = 0.0
        f_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader):

            # get the inputs
            if self.phase == SUPERVISED_TRAINING:
                x, y = data[0].cuda(), data[1].cuda()
            else:
                x = data.cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            if self.phase == SUPERVISED_TRAINING:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred, x)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            if self.phase == SUPERVISED_TRAINING:
                # compute other interesting metrics
                y_ = F.softmax(y_pred, dim=1).data.cpu().numpy()[:,1,...]
                j_cum += jaccard(y_, y.cpu().numpy())
                a, p, r, f = accuracy_metrics(y_, y.cpu().numpy())
                a_cum += a; p_cum += p; r_cum += r; f_cum += f

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        j_avg = j_cum / cnt
        a_avg = a_cum / cnt
        p_avg = p_cum / cnt
        r_avg = r_cum / cnt
        f_avg = f_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            if self.phase == SUPERVISED_TRAINING:
                writer.add_scalar('test/loss-seg', loss_avg, epoch)
                writer.add_scalar('test/jaccard', j_avg, epoch)
                writer.add_scalar('test/accuracy', a_avg, epoch)
                writer.add_scalar('test/precision', p_avg, epoch)
                writer.add_scalar('test/recall', r_avg, epoch)
                writer.add_scalar('test/f-score', f_avg, epoch)
                if write_images:
                    # write images
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                    y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                    writer.add_image('test/x', x, epoch)
                    writer.add_image('test/y', y, epoch)
                    writer.add_image('test/y_pred', y_pred, epoch)
            else:
                writer.add_scalar('test/loss-rec', loss_avg, epoch)
                if write_images:
                    x = vutils.make_grid(x, normalize=True, scale_each=True)
                    x_rec = vutils.make_grid(torch.sigmoid(y_pred).data)
                    writer.add_image('test/x-rec-input', x, epoch)
                    writer.add_image('test/x-rec-output', x_rec, epoch)

        return loss_avg

    # trains the network
    def train_net(self, train_loader, test_loader, loss_fn_seg, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1,
                  train_loader_unsupervised=None, test_loader_unsupervised=None, loss_fn_rec=None):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        if self.pretrain_unsupervised:

            print('[%s] Starting unsupervised pre-training' % (datetime.datetime.now()))

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader_unsupervised, loss_fn=loss_fn_rec, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader_unsupervised, loss_fn=loss_fn_rec, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint_rec.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch_rec'))

            print('[%s] Starting supervised pre-training' % (datetime.datetime.now()))
            self.phase = SUPERVISED_TRAINING

            test_loss_min = np.inf
            for epoch in range(epochs):

                print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

                # train the model for one epoch
                self.train_epoch(loader=train_loader, loss_fn=loss_fn_seg, optimizer=optimizer, epoch=epoch,
                                 print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

                # adjust learning rate if necessary
                if scheduler is not None:
                    scheduler.step(epoch=epoch)

                    # and keep track of the learning rate
                    writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

                # test the model for one epoch is necessary
                if epoch % test_freq == 0:
                    test_loss = self.test_epoch(loader=test_loader, loss_fn=loss_fn_seg, epoch=epoch, writer=writer, write_images=True)

                    # and save model if lower test loss is found
                    if test_loss < test_loss_min:
                        test_loss_min = test_loss
                        torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

                # save model every epoch
                torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

            writer.close()