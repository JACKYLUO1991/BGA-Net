import argparse
import time
import numpy as np

import torch
# from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3 import *

from utils.evaluation_metrics_for_segmentation import evaluate_segmentation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/refuge_weights.tar',
                        help='Model path')
    parser.add_argument(
        '--dataset', type=str, default='Drishti-GS', help='test folder id contain images ROIs to test'
    )
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument(
        '--resize', type=int, default=800, help='image resize')

    parser.add_argument(
        '--data-dir',
        default='./fundus/',
        help='data root path'
    )
    parser.add_argument(
        '--mask-dir',
        required=True,
        default='./fundus/Drishti-GS/test/mask',
        help='mask image path'
    )
    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
    )
    parser.add_argument(
        '--save-root-ent',
        type=str,
        default='./results/ent/',
        help='path to save ent',
    )
    parser.add_argument(
        '--save-root-mask',
        type=str,
        default='./results/mask/',
        help='path to save mask',
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
    parser.add_argument('--test-prediction-save-path', type=str,
                        default='./results/baseline/',
                        help='Path root for test image and mask')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Scale(args.resize),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    split_data = 'testval'
    if args.dataset == 'refuge':
        split_data = 'test'
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split=split_data,
                                    transform=composed_transforms_test)

    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    # try:
    # model.load_state_dict(checkpoint)
    # pretrained_dict = checkpoint['model_state_dict']
    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)

    # except Exception:
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        raise FileNotFoundError('No checkpoint file exist...')

    model.eval()
    print('==> Evaluating with %s' % args.dataset)

    test_cup_dice = 0.0
    test_disc_dice = 0.0
    timestamp_start = \
        datetime.now(pytz.timezone('Asia/Hong_Kong'))

    durings = []
    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                             total=len(test_loader),
                                             ncols=80, leave=False):
            data = sample['image']
            target = sample['map']
            img_name = sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)

            torch.cuda.synchronize()
            start = time.time()
            prediction, boundary = model(data)
            prediction = torch.nn.functional.interpolate(prediction, size=(target.size()[2], target.size()[3]),
                                                         mode="bilinear")
            boundary = torch.nn.functional.interpolate(boundary, size=(target.size()[2], target.size()[3]),
                                                       mode="bilinear")
            data = torch.nn.functional.interpolate(data, size=(target.size()[2], target.size()[3]), mode="bilinear")
            torch.cuda.synchronize()
            end = time.time()
            during = end - start
            durings.append(during)

            cup_dice, disc_dice = dice_coeff_2label(prediction, target)
            test_cup_dice += cup_dice
            test_disc_dice += disc_dice

            # drawing figures
            draw_ent(torch.sigmoid(prediction).data.cpu()[0].numpy(), os.path.join(args.save_root_ent, args.dataset),
                     img_name[0])
            draw_mask(torch.sigmoid(prediction).data.cpu()[0].numpy(), os.path.join(args.save_root_mask, args.dataset),
                      img_name[0])
            draw_boundary(torch.sigmoid(boundary).data.cpu()[0].numpy(),
                          os.path.join(args.save_root_mask, args.dataset), img_name[0])

            prediction, ROI_mask = postprocessing(torch.sigmoid(prediction).data.cpu()[0], dataset=args.dataset)
            imgs = data.data.cpu()
            target_numpy = target.cpu().numpy()

            for img, lt, lp in zip(imgs, target_numpy, [prediction]):
                img, lt = untransform(img, lt)
                save_per_img(img.numpy().transpose(1, 2, 0), os.path.join(args.test_prediction_save_path, args.dataset),
                             img_name[0], lp, lt, ROI_mask)

        test_cup_dice /= len(test_loader)
        test_disc_dice /= len(test_loader)

        print("test_cup_dice = ", test_cup_dice)
        print("test_disc_dice = ", test_disc_dice)

    durings = durings[:-1]
    tim = np.sum(durings) / len(durings)
    print("Time: ", tim)


    # submit script
    _, _, mae_cdr, _ = evaluate_segmentation_results(
        osp.join(args.test_prediction_save_path, args.dataset, 'pred_mask'),
        args.mask_dir, output_path="./results", export_table=True)

    with open(osp.join(args.test_prediction_save_path, 'test_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = [[args.model_file] + ['cup dice: '] + \
               [test_cup_dice] + ['disc dice: '] + \
               [test_disc_dice] + ['cdr: '] + \
               [mae_cdr] + [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
