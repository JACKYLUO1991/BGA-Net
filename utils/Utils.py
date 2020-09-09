import os.path as osp
import os
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
from skimage.measure import label, regionprops
from skimage import measure

import matplotlib.pyplot as plt

plt.switch_backend('agg')

from utils.metrics import *
import cv2


def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    """
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    """
    min = np.amin(ent)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    """
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    """
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_boundary(prediction, save_root, name):
    """
    Draw the mask probability for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    """
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]
    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0

    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def postprocessing(prediction, threshold=0.75, dataset='G'):
    if dataset[0] == 'D':
        prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # _disc_mask = disc_mask.copy()
        # _cup_mask = cup_mask.copy()
        for i in range(5):
            disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
            cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)
        cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask

        # ROI_mask = _cup_mask + _disc_mask
        ROI_mask = cup_mask + disc_mask
        ROI_mask[ROI_mask < 1] = 255
        ROI_mask[ROI_mask < 2] = 128
        ROI_mask[ROI_mask < 3] = 0

        return prediction_copy, ROI_mask
    else:
        prediction = prediction.numpy()
        prediction = (prediction > threshold)  # return binary mask
        prediction = prediction.astype(np.uint8)
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        # _disc_mask = disc_mask.copy()
        # _cup_mask = cup_mask.copy()
        for i in range(5):
            disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
            cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)
        cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask

        # ROI_mask = _cup_mask + _disc_mask
        ROI_mask = cup_mask + disc_mask
        ROI_mask[ROI_mask < 1] = 255
        ROI_mask[ROI_mask < 2] = 128
        ROI_mask[ROI_mask < 3] = 0

        return prediction_copy, ROI_mask


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:, :, 0] = mask[0]
    _mask[:, :, 1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)

    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0]: (i + 1) * img_shape[0], :, :] = img[i]
    imsave(name, stack_image)


def save_per_img(patch_image, data_save_path, img_name, prob_map, boundary_map, ROI_mask):
    path3 = os.path.join(data_save_path, 'pred_mask', img_name.split('.')[0] + '.bmp')
    path2 = os.path.join(data_save_path, 'gt_overlay', img_name.split('.')[0] + '.png')
    path1 = os.path.join(data_save_path, 'pred_overlay', img_name.split('.')[0] + '.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0] + '.png')

    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))
    if not os.path.exists(os.path.dirname(path2)):
        os.makedirs(os.path.dirname(path2))
    if not os.path.exists(os.path.dirname(path3)):
        os.makedirs(os.path.dirname(path3))
        
    # save mask output
    Image.fromarray(ROI_mask).save(path3)

    disc_map = prob_map[0]
    cup_map = prob_map[1]
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0])
    disc_map[:, size[1] - 1] = np.zeros(size[0])
    disc_map[0, :] = np.zeros(size[1])
    disc_map[size[0] - 1, :] = np.zeros(size[1])
    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])

    disc_mask = (disc_map > 0.75)  # return binary mask
    cup_mask = (cup_map > 0.75)
    disc_mask = disc_mask.astype(np.uint8)
    cup_mask = cup_mask.astype(np.uint8)

    for i in range(5):
        disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        cup_mask = scipy.signal.medfilt2d(cup_mask, 7)

    disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    disc_mask = get_largest_fillhole(disc_mask)
    cup_mask = get_largest_fillhole(cup_mask)

    disc_mask = morphology.binary_dilation(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    cup_mask = morphology.binary_dilation(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1

    disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
    cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)

    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)

    patch_image3 = patch_image.astype(np.uint8).copy()
    patch_image3 = Image.fromarray(patch_image3)
    patch_image3.save(path0)

    # **********************************  ground truth boundary process   **********************************#
    patch_image2 = patch_image.astype(np.uint8).copy()

    b_cup_map = boundary_map[1].astype(np.uint8)
    b_optic_map = boundary_map[0].astype(np.uint8)

    contours_cup_T = measure.find_contours(b_cup_map, 0.5)
    contours_disc_T = measure.find_contours(b_optic_map, 0.5)

    for n, contour in enumerate(contours_cup_T):
        patch_image2[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image2[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image2[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image2[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image2[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image2[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image2[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc_T):
        patch_image2[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image2[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image2[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image2[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image2[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image2[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image2[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    patch_image2 = Image.fromarray(patch_image2)
    patch_image2.save(path2)
    # **********************************  ground truth boundary process   **********************************#

    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)
    patch_image.save(path1)


def untransform(img, lt):
    img = (img + 1) * 127.5
    # lt = lt * 128  # useless
    lt = lt
    return img, lt
