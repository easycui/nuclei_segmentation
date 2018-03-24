from __future__ import print_function
import h5py
import sys
import os
import numpy as np
from PIL import Image
import ConfigParser
import math


from matplotlib import pyplot as plt


# color : red, yellow, green, blue
def get_color(category):
    # category can be 2, 3, 4
    # Two Category is black-white
    if category == 2:
        return [(0, 0, 0), (255, 255, 255)]
    # Three Category is red-green-blue
    elif category == 3:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Four category is red-green-blue-yello
    else:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 0)]


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def get_dataset(imgs_dir, gt_dir, masks_dir, Nimgs, height, width, channels, category):
    imgs = np.empty((Nimgs, height, width, channels))
    gt = np.empty((Nimgs, height, width))
    border_masks = np.empty((Nimgs, height, width))
    if not os.walk(imgs_dir):
        print("directory don't exist")
        sys.exit()
    print("-----------------------------------------------------------------")
    print('read images')
    for i in range(Nimgs):  # list all files, directories in the path
        img_file = str(i+1) + '_sample.bmp'
        # original
        print("original image: ", img_file)
        img = Image.open(imgs_dir + img_file)
        #img = np.asarray(img)
        #print(img.shape)
        imgs[i] = np.reshape(np.asarray(img), (height, width, channels))

        # corresponding ground truth

        gt_file = str(i+1) + "_gt.bmp"
        print("ground truth name: ", gt_file)
        g_truth = Image.open(gt_dir + gt_file)
        gt[i] = np.asarray(g_truth)

        # corresponding border masks
        mask_file = str(i+1) + "_mask.bmp"
        print("border masks name: ", mask_file)
        b_mask = Image.open(masks_dir + mask_file)
        border_masks[i] = np.asarray(b_mask)

    print("imgs max: {}, min: {}".format(np.max(imgs), np.min(imgs)))
    assert (np.max(gt) == category - 1 and np.max(border_masks) == 255)
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (Nimgs, channels, height, width))
    gt = np.reshape(gt, (Nimgs, 1, height, width))
    assert (gt.shape == (Nimgs, 1, height, width))
    border_masks = np.reshape(border_masks, (Nimgs, 1, height, width))

    assert (border_masks.shape == (Nimgs, 1, height, width))
    return imgs, gt, border_masks


# convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


# prepare the gt/target in the right shape for the Unet
# transform from label to one hot array
def get_gt(gt, category_num):
    assert (len(gt.shape) == 4)  # 4D arrays
    assert (gt.shape[1] == 1)  # check the channel is 1
    im_h = gt.shape[2]
    im_w = gt.shape[3]
    gt = np.reshape(gt, (gt.shape[0], im_h * im_w))
    new_gt = np.zeros((gt.shape[0], im_h * im_w, category_num), dtype=np.uint8)

    for label in range(category_num):
        one_label = new_gt[:,:,label]
        one_label[np.where(gt == label)] = 1
    return new_gt

def get_loss_weight(patch_height, patch_width, mode, stride_height=8, stride_width=8, batch_size=4, border = 16):
    loss_weight = np.zeros((patch_height, patch_width))
    center_x = patch_height /2 - 1
    center_y = patch_width / 2 - 1
    if mode == 0:
        return None

    for k in range(patch_height//2):
        for i in range(k, patch_width - k):
            loss_weight[k, i] = k
            loss_weight[i, k] = k
            loss_weight[patch_height - k - 1, i] = k
            loss_weight[i, patch_width - k - 1] = k
    max_value = np.max(loss_weight)
    max_value = float(max_value)
    if mode == 4:
        # in this mode, loss weight outside is 0, inner is 1
        loss_weight[np.where(loss_weight < border)] = 0
        loss_weight[np.where(loss_weight >= border)] = 1
        loss_weight = np.reshape(loss_weight, (patch_width * patch_height))
    else:
        if mode == 1:
            loss_weight = loss_weight/max_value * loss_weight/max_value
        elif mode == 2:
            loss_weight = loss_weight/max_value
        elif mode == 3:
            loss_weight = np.sqrt(loss_weight/max_value)

        loss_weight = np.reshape(loss_weight, (patch_width * patch_height))
        weight_sum = patch_height * patch_width
        cur_sum = np.sum(loss_weight)
        loss_weight *= weight_sum/cur_sum
    #     loss_weight = np.reshape(loss_weight[:,:,0], (patch_width * patch_height))
    #     loss_weight += 0.01
    #     weight_sum = patch_height * patch_width
    #     cur_sum = np.sum(loss_weight)
    #     loss_weight *= weight_sum/cur_sum

        #loss_weight = np.reshape(loss_weight[:,0], (patch_width*patch_height,1))
    result = loss_weight
    print("shape of loss_weight:", result.shape)
    return result


def pred_to_imgs(pred, real_value=1):
    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,category_num)
    category_num = pred.shape[2]
    patch_height = int(np.sqrt(pred.shape[1]))
    patch_width = int(np.sqrt(pred.shape[1]))
    print("category number is:", category_num)
    #pred_images = np.empty((pred.shape[0], pred.shape[1]))  # (Npatches,height*width)
    if real_value == 0:
        #for i in range(pred.shape[0]):
            #for pix in range(pred.shape[1]):
                #pred_images[i, pix] = np.argmax(pred[i, pix, :])
        pred_images = np.argmax(pred, category_num)

    else:
        if category_num == 2:
            pred_images = pred[:,:,1]
        if category_num == 3:
            pred_images = pred
            pred_images = np.transpose(pred_images, (0,2,1))
    pred_images = np.reshape(pred_images, (pred_images.shape[0], category_num, patch_height, patch_width))
    return pred_images


def label2rgb(imgs, category):
    if category > 4:
        print("ERROR: at most 4 categories")
        exit()
    assert (len(imgs.shape) == 4)
    result = np.zeros((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
    color = get_color(category)
    for k in range(imgs.shape[0]):
        for i in range(imgs.shape[2]):
            for j in range(imgs.shape[3]):
                c = int(imgs[k, 0, i, j])
                # 3 channels
                for m in range(3):
                    result[k, m, i, j] = color[c][m]
    return result


def parse_config(configure_file):
    config_content = {}
    config = ConfigParser.RawConfigParser()
    config.read(configure_file)
    print(config)
    # ------------the dataset path----------------------------------------------------------------
    dataset = config.get('dataset name', 'dataset')
    config_content['dataset'] = dataset
    dataset_path = dataset + '_datasets_hdf5/'
    config_content['dataset_path'] = dataset_path

    # ------------the file path of Hdf5 files for test and train data-----------------------------
    for items in config.items('hdf5 files'):
        config_content[items[0]] = dataset_path + items[1]
    # ------------image properties----------------------------------------------------------------
    for items in config.items('image properties'):
        config_content[items[0]] = int(items[1])
    # -------------experiment name----------------------------------------------------------------
    name = config.get('experiment name', 'name')
    config_content['name'] = name
    net = config.get('experiment name', 'net')
    config_content['net'] = net
    # -------------data attributes----------------------------------------------------------------
    for items in config.items('data attributes'):
        config_content[items[0]] = int(items[1])
    # ------------Path of the images --------------------------------------------------------------
    for items in config.items('image path'):
        config_content[items[0]] = dataset + items[1]
    # ------------train setting---------------------------------------
    for items in config.items('training settings'):
        config_content[items[0]] = int(items[1])
    # ------------test setting----------------------------------------
    for items in config.items('testing settings'):
        config_content[items[0]] = int(items[1])

    for items in config.items('others'):
        config_content[items[0]] = int(items[1])

    return config_content
