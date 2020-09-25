from datetime import datetime
import timeit
from torchvision.utils import make_grid

import pytz
import torch.nn.functional as F

from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import tqdm
import socket
from utils.metrics import dice_coeff_2label
from utils.Utils import *

import numpy as np

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


class Trainer(object):

    def __init__(self, cuda, model_gen, model_bd, model_mask, optimizer_gen, optim_bd,
                 optim_mask, train_loader, validation_loader, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=10,
                 coefficient=0.0, boundary_exist=True):
        self.cuda = cuda
        self.warmup_epoch = warmup_epoch
        self.model_gen = model_gen  # generator
        self.model_bd = model_bd  # boundary adversarial
        self.model_mask = model_mask  # mask adversarial
        self.optim_gen = optimizer_gen
        self.optim_bd = optim_bd
        self.optim_mask = optim_mask
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size
        self.coefficient = coefficient

        # whether or not using boundary segmentation branch
        self.bd_weight = 0
        if boundary_exist:
            self.bd_weight = 1

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = 10
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/loss_adv',
            'train/loss_disc',
            'valid/loss_seg',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_seg_loss = 0.0
        self.running_adv_loss = 0.0
        self.running_disc_loss = 0.0
        # self.best_cup_dice = 0.0
        self.best_epoch = -1
        self.best_loss = np.inf

    def validate(self):
        training = self.model_gen.training
        self.model_gen.eval()

        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        metrics = []

        with torch.no_grad():
            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.validation_loader), total=len(self.validation_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['map']
                target_boundary = sample['boundary']
                if self.cuda:
                    data, target_map, target_boundary = data.cuda(), target_map.cuda(), target_boundary.cuda()
                # with torch.no_grad():    # redundant
                predictions, boundary = self.model_gen(data)

                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                val_loss += loss_data

                # dice coefficient calculation
                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
                val_cup_dice += dice_cup
                val_disc_dice += dice_disc

            val_loss /= len(self.validation_loader)
            val_cup_dice /= len(self.validation_loader)
            val_disc_dice /= len(self.validation_loader)
            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            metrics = np.mean(metrics, axis=0)

            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.validation_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.validation_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.validation_loader)))

            mean_dice = (val_cup_dice * 2 + val_disc_dice) / 3  # cup weight higher than disc
            is_best = mean_dice > self.best_disc_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_disc_dice = mean_dice
            # is_best = val_loss < self.best_loss
            # if is_best:
            #     self.best_epoch = self.epoch + 1
            #     self.best_loss = val_loss

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_bd_state_dict': self.optim_bd.state_dict(),
                    'optim_mask_state_dict': self.optim_mask.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_bd_state_dict': self.model_bd.state_dict(),
                    'model_mask_state_dict': self.model_mask.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_bd': get_lr(self.optim_bd),
                    'learning_rate_mask': get_lr(self.optim_mask),
                    # 'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            # else:
            #     if (self.epoch + 1) % 50 == 0:
            #         torch.save({
            #             'epoch': self.epoch,
            #             'iteration': self.iteration,
            #             'arch': self.model_gen.__class__.__name__,
            #             'optim_state_dict': self.optim_gen.state_dict(),
            #             'optim_bd_state_dict': self.model_bd.state_dict(),
            #             'optim_mask_state_dict': self.model_gen.state_dict(),
            #             'model_state_dict': self.model_gen.state_dict(),
            #             'model_bd_state_dict': self.model_bd.state_dict(),
            #             'model_mask_state_dict': self.model_mask.state_dict(),
            #             'learning_rate_gen': get_lr(self.optim_gen),
            #             'learning_rate_bd': get_lr(self.optim_bd),
            #             'learning_rate_mask': get_lr(self.optim_mask),
            #             # 'best_mean_dice': self.best_mean_dice,
            #         }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.now(pytz.timezone(self.time_zone)) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 3 + metrics.tolist() + [elapsed_time] + [
                    'best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.validation_loader)))

            if training:
                self.model_gen.train()
                self.model_bd.train()
                self.model_mask.train()

    def train_epoch(self):
        real_label = 1
        fake_label = 0

        # smooth = 1e-7
        self.model_gen.train()
        self.model_bd.train()
        self.model_mask.train()
        # self.running_seg_loss = 0.0
        # self.running_adv_loss = 0.0
        # self.running_total_loss = 0.0
        # self.running_cup_dice_tr = 0.0
        # self.running_disc_dice_tr = 0.0
        # loss_adv_diff_data = 0
        # loss_D_same_data = 0
        # loss_D_diff_data = 0

        loss_seg_data = 0
        loss_adv_data = 0
        loss_disc_data = 0

        max_iteration = self.stop_epoch * len(self.train_loader)
        # validation_loader = enumerate(self.validation_loader)
        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            metrics = []
            iteration = batch_idx + self.epoch * len(self.train_loader)
            # 'ploy' learning rate
            _ = adjust_learning_rate(self.optim_gen, self.lr_gen, iteration, max_iteration, 0.9)
            self.iteration = iteration

            # assert self.model_gen.training
            # assert self.model_dis.training
            # assert self.model_dis2.training

            self.optim_gen.zero_grad()
            self.optim_bd.zero_grad()
            self.optim_mask.zero_grad()

            # 1. train generator with random images
            for param in self.model_bd.parameters():
                param.requires_grad = False
            for param in self.model_mask.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = True

            imageS = sample['image'].cuda()
            target_map = sample['map'].cuda()
            target_boundary = sample['boundary'].cuda()

            oS, boundaryS = self.model_gen(imageS)

            loss_seg1 = bceloss(torch.sigmoid(oS), target_map)  # mask
            loss_seg2 = mseloss(torch.sigmoid(boundaryS), target_boundary)  # boundary
            loss_seg = loss_seg1 + loss_seg2 * self.bd_weight

            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            loss_seg.backward(retain_graph=True)

            if self.epoch > self.warmup_epoch:
                # 2. train generator with images from different domain
                # try:
                #     id_, sampleT = next(validation_loader)
                # except:
                #     domain_t_loader = enumerate(self.validation_loader)
                #     id_, sampleT = next(domain_t_loader)

                # imageT = sampleT['image'].cuda()
                # target_map = sample['map'].cuda()
                # target_boundary = sample['boundary'].cuda()
                #
                # oT, boundaryT = self.model_gen(imageT)
                # uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)  # Shannon Entropy
                # D_out2 = self.model_dis(torch.sigmoid(boundaryT))  # boundary-driven
                # D_out1 = self.model_dis2(uncertainty_mapT)  # entropy-driven
                #
                # loss_adv_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                #     source_domain_label).cuda())
                # loss_adv_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                #     source_domain_label).cuda())
                # loss_adv_diff = 0.01 * (loss_adv_diff1 + loss_adv_diff2)
                # self.running_adv_diff_loss += loss_adv_diff.item()
                # loss_adv_diff_data = loss_adv_diff.data.item()  # the same to loss_adv_diff.item()
                # if np.isnan(loss_adv_diff_data):
                #     raise ValueError('loss_adv_diff_data is nan while training')

                boundaryS_A = self.model_bd(torch.sigmoid(boundaryS))
                mask_A = self.model_mask(torch.sigmoid(oS))
                loss_adv_bd = F.binary_cross_entropy_with_logits(boundaryS_A,
                                                                 torch.FloatTensor(boundaryS_A.data.size()).fill_(
                                                                     real_label).cuda())
                loss_adv_mask = F.binary_cross_entropy_with_logits(mask_A, torch.FloatTensor(mask_A.data.size()).fill_(
                    real_label).cuda())

                loss_adv = self.coefficient * (loss_adv_bd + loss_adv_mask)
                self.running_adv_loss += loss_adv.item()
                loss_adv_data = loss_adv.data.item()
                loss_adv.backward()
                self.optim_gen.step()

                # 3. train discriminator with images from same domain
                for param in self.model_bd.parameters():
                    param.requires_grad = True
                for param in self.model_mask.parameters():
                    param.requires_grad = True
                for param in self.model_gen.parameters():
                    param.requires_grad = False

                # boundaryS = boundaryS.detach()  # pure tensor, no gradient
                # oS = oS.detach()
                # # uncertainty_mapS = -1.0 * torch.sigmoid(oS) * torch.log(torch.sigmoid(oS) + smooth)
                # D_boundary = self.model_bd(torch.sigmoid(boundaryS))
                # D_map = self.model_gen(torch.sigmoid(oS))
                #
                # loss_D_bd = F.binary_cross_entropy_with_logits(D_boundary,
                #                                                torch.FloatTensor(D_boundary.data.size()).fill_(
                #                                                    source_domain_label).cuda())
                # loss_D_mask = F.binary_cross_entropy_with_logits(D_map, torch.FloatTensor(D_map.data.size()).fill_(
                #     source_domain_label).cuda())
                # loss_D_same = loss_D_bd + loss_D_mask
                #
                # self.running_dis_same_loss += loss_D_same.item()
                # loss_D_same_data = loss_D_same.data.item()
                # loss_D_same.backward()

                # 4. train discriminator with images from different domain
                boundaryS_D = boundaryS.detach()
                oS_D = oS.detach()
                boundary_D = target_boundary.detach()
                o_D = target_map.detach()

                # uncertainty_mapT = -1.0 * torch.sigmoid(oT) * torch.log(torch.sigmoid(oT) + smooth)
                D_bd = self.model_bd(torch.sigmoid(boundaryS_D))
                D_mask = self.model_mask(torch.sigmoid(oS_D))

                D_db_T = self.model_bd(boundary_D)
                D_mask_T = self.model_mask(o_D)

                loss_D_bd = F.binary_cross_entropy_with_logits(D_bd, torch.FloatTensor(D_bd.data.size()).fill_(
                    fake_label).cuda())
                loss_D_mask = F.binary_cross_entropy_with_logits(D_mask, torch.FloatTensor(D_mask.data.size()).fill_(
                    fake_label).cuda())

                loss_D_bd_T = F.binary_cross_entropy_with_logits(D_db_T, torch.FloatTensor(D_db_T.data.size()).fill_(
                    real_label).cuda())
                loss_D_mask_T = F.binary_cross_entropy_with_logits(D_mask_T,
                                                                   torch.FloatTensor(D_mask_T.data.size()).fill_(
                                                                       real_label).cuda())

                loss_disc = self.coefficient * (loss_D_bd + loss_D_mask + loss_D_mask_T + loss_D_bd_T)
                self.running_disc_loss += loss_disc.item()
                loss_disc_data = loss_disc.data.item()
                loss_disc.backward()

                # 5. update parameters
                self.optim_bd.step()
                self.optim_mask.step()

                # segmentation map visualization
                if iteration % 30 == 0:
                    grid_image = make_grid(
                        imageS[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/image', grid_image, iteration)
                    grid_image = make_grid(
                        target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/target_cup', grid_image, iteration)
                    grid_image = make_grid(
                        target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/target_disc', grid_image, iteration)
                    grid_image = make_grid(
                        target_boundary[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/target_boundary', grid_image, iteration)
                    grid_image = make_grid(torch.sigmoid(oS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/prediction_cup', grid_image, iteration)
                    grid_image = make_grid(torch.sigmoid(oS)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/prediction_disc', grid_image, iteration)
                    grid_image = make_grid(torch.sigmoid(boundaryS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('DomainS/prediction_boundary', grid_image, iteration)

                self.writer.add_scalar('train_adv/loss_adv', loss_adv_data, iteration)
                self.writer.add_scalar('train_adv/loss_disc', loss_disc_data, iteration)

            self.writer.add_scalar('train_gen/loss_seg', loss_seg_data, iteration)

            metrics.append((loss_seg_data, loss_adv_data, loss_disc_data))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.now(pytz.timezone(self.time_zone)) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + metrics.tolist() + [''] * 3 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        # self.running_seg_loss /= len(self.domain_loaderS)
        # self.running_adv_diff_loss /= len(self.domain_loaderS)
        # self.running_dis_same_loss /= len(self.domain_loaderS)
        # self.running_dis_diff_loss /= len(self.domain_loaderS)

        self.running_seg_loss /= len(self.train_loader)
        self.running_adv_loss /= len(self.train_loader)
        self.running_disc_loss /= len(self.train_loader)

        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
              ' Average advLoss: %f, Average disLoss: %f, '
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
               self.running_adv_loss, self.running_disc_loss, stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()  # one epoch
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            # if (epoch + 1) % 100 == 0:
            #     _lr_gen = self.lr_gen * self.lr_decrease_rate
            #     for param_group in self.optim_gen.param_groups:
            #         param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_gen), self.epoch * (len(self.train_loader)))

            if (self.epoch + 1) % self.interval_validate == 0:
                self.validate()
        self.writer.close()
