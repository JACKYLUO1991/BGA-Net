from datetime import datetime
import os
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.GAN import BoundaryDiscriminator, MaskDiscriminator

here = osp.dirname(osp.abspath(__file__))


def main():
    # Add default values to all parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument(
        '--coefficient', type=float, default=0.01, help='balance coefficient'
    )
    parser.add_argument(
        '--boundary-exist', type=bool, default=True, help='whether or not using boundary branch'
    )
    parser.add_argument(
        '--dataset', type=str, default='refuge', help='folder id contain images ROIs to train or validation'
    )
    parser.add_argument(
        '--batch-size', type=int, default=12, help='batch size for training the model'
    )
    # parser.add_argument(
    #     '--group-num', type=int, default=1, help='group number for group normalization'
    # )
    parser.add_argument(
        '--max-epoch', type=int, default=300, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=300, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
    )
    parser.add_argument(
        '--interval-validate', type=int, default=1, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-5, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default='./fundus/',
        help='data root path'
    )
    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
    )
    parser.add_argument(
        '--sync-bn',
        type=bool,
        default=False,
        help='sync-bn in deeplabv3+',
    )
    parser.add_argument(
        '--freeze-bn',
        type=bool,
        default=False,
        help='freeze batch normalization of deeplabv3+',
    )

    args = parser.parse_args()
    args.model = 'MobileNetV2'

    now = datetime.now()
    args.out = osp.join(here, 'logs', args.dataset, now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)

    # save training hyperparameters or/and settings
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(2020)
    if cuda:
        torch.cuda.manual_seed(2020)

    # 1. loading data
    composed_transforms_train = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_val = transforms.Compose([
        tr.RandomCrop(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    data_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train',
                                       transform=composed_transforms_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    data_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test',
                                     transform=composed_transforms_val)
    dataloader_val = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
    #                                    transform=composed_transforms_ts)
    # domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2,
    #                                pin_memory=True)

    # 2. model
    model_gen = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                        sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    model_bd = BoundaryDiscriminator().cuda()
    model_mask = MaskDiscriminator().cuda()

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    optim_gen = torch.optim.Adam(
        model_gen.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.99)
    )
    optim_bd = torch.optim.SGD(
        model_bd.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optim_mask = torch.optim.SGD(
        model_mask.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # breakpoint recovery
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_gen.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model_gen.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_bd_state_dict']
        model_dict = model_bd.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_bd.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_mask_state_dict']
        model_dict = model_mask.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_mask.load_state_dict(model_dict)

        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        optim_gen.load_state_dict(checkpoint['optim_state_dict'])
        optim_bd.load_state_dict(checkpoint['optim_bd_state_dict'])
        optim_mask.load_state_dict(checkpoint['optim_mask_state_dict'])

    trainer = Trainer.Trainer(
        cuda=cuda,
        model_gen=model_gen,
        model_bd=model_bd,
        model_mask=model_mask,
        optimizer_gen=optim_gen,
        optim_bd=optim_bd,
        optim_mask=optim_mask,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=dataloader_train,
        validation_loader=dataloader_val,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch,
        coefficient=args.coefficient,
        boundary_exist=args.boundary_exist
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
