from __future__ import print_function
import numpy as np
import random
import ConfigParser
import sys
import help_functions as hf


# To select the same images
# random.seed(10)

# Load the original data and return the extracted patches for training

def get_data_training(train_imgs_file,
                      train_gt_file,
                      patch_height,
                      patch_width,
                      num_patches,
                      channel,
                      config):
    """
    :param train_imgs_file:  the filename of training images
    :param train_gt_file: the filename of training ground truth images
    :param patch_height: the height of patch
    :param patch_width: the width of patch
    :param num_patches: the number of patches need to be generated
    :return:
    """
    train_images = hf.load_hdf5(train_imgs_file)
    train_gt = hf.load_hdf5(train_gt_file)
    # visualize(group_images(train_imgs_file[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    #train_images = pp.my_PreProc(train_images, config)

    print("shape of training gt sample", train_gt.shape)
    data_consistency_check(train_images, train_gt)

    # The the number of categories
    # print np.shape(np.unique(train_gt))
    category_num = np.shape(np.unique(train_gt))[0]
    print("----------------------------------------------")
    print("shape of training image sample:", train_images.shape)
    print('----------------------------------------------')
    print("train images range (min-max): {}- {} "
          .format(np.min(train_images), np.max(train_images)))

    # randomly extract the TRAINING patches from the full images
    if config['random'] == 1:
        patches_imgs_train, patches_gt_train = extract_random(train_images, train_gt,
                                                              patch_height, patch_width,
                                                              num_patches, channel)
    else:
        patches_imgs_train, patches_gt_train = extract_ordered(train_images, train_gt,
                                                              patch_height, patch_width,
                                                              channel)
    data_consistency_check(patches_imgs_train, patches_gt_train)

    # print some information of the data

    print('---------------------patches info------------------------')
    print('category_num', category_num)
    print("train PATCHES images shape:", patches_imgs_train.shape)
    print("train PATCHES images range: {}- {}, data type: {}"
          .format(np.min(patches_imgs_train), np.max(patches_imgs_train), patches_imgs_train.dtype))
    print('gt patch value range {} - {}, data type: {}'
          .format(np.min(patches_gt_train), np.max(patches_gt_train), patches_gt_train.dtype))
    print('---------------------------------------------------------')
    return patches_imgs_train, patches_gt_train, category_num


# Load the original data and return the extracted patches for testing
# without overlap
def get_data_testing(test_images_file, test_gt_file, n_test_images, patch_height, patch_width, channel, config):

    test_imgs_original = hf.load_hdf5(test_images_file)
    test_gt = hf.load_hdf5(test_gt_file)

    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0: n_test_images, :, :, :]
    test_gt = test_gt[0: n_test_images, :, :, :]
    test_imgs = paint_border(test_imgs, patch_height, patch_width, channel)
    test_gt = paint_border(test_gt, patch_height, patch_width, channel)
    data_consistency_check(test_imgs, test_gt)

    # check masks are within 0-1
    assert np.min(test_gt) == 0
    print('-----------------------------------------------------------')
    print("test images/masks shape:", test_imgs.shape)
    print("test images range (min-max):{}-{} ".format(np.min(test_imgs), np.max(test_imgs)))
    print("test masks are within 0-1\n")
    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs, patch_height, patch_width)
    print("test PATCHES images/masks shape:", patches_imgs_test.shape)
    print("test PATCHES images range (min-max): {}-{} "
          .format(np.min(patches_imgs_test), np.max(patches_imgs_test)))

    print('-----------------------------------------------------------')
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_gt, np.min(test_gt) + 1


# Load the original data and return the extracted patches for testing
# with overlop
# return the ground truth in its original shape
def get_data_testing_overlap(test_images_file, test_gt_file, n_test_images, patch_height, patch_width,
                             stride_height, stride_width, channel, config):
    """
    :param test_images_file: the filename of hdf5 test_images_file
    :param test_gt_file: the filename of hdf5 test_gt_file
    :param n_test_images: the num of test image
    :param patch_height: the height of each patch
    :param patch_width: the width of each width
    :param stride_height: the stride of height
    :param stride_width: the stride of width
    :return:
    """

    test_images = hf.load_hdf5(test_images_file)
    test_gt = hf.load_hdf5(test_gt_file)
    # preproceing the test images
    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_images = test_images[0:n_test_images, :, :, :]
    test_gt = test_gt[0:n_test_images, :, :, :]
    test_images = paint_border_overlap(test_images, patch_height, patch_width,
                                       stride_height, stride_width, channel)

    print("extended test images shape:", test_images.shape)
    print("print sample data:", test_images[0, 0, 0:100, 0:100])
    print("test ground truth shape:", test_gt.shape)
    print("sample gt:", test_gt[0, 0, 0:100, 0:100])

    print("test images range (min-max): {}-{} ".format(np.min(test_images), np.max(test_images)))

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_images, patch_height, patch_width,
                                                stride_height, stride_width, channel)

    print("test PATCHES images shape:", patches_imgs_test.shape)
    print("test PATCHES images range (min-max):{} - {}"
          .format(np.min(patches_imgs_test), np.max(patches_imgs_test)))

    return patches_imgs_test, test_images.shape[2], test_images.shape[3], test_gt, np.max(test_gt) + 1


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim

    # extend dimension of img h by adding zeros
    if leftover_h != 0:
        print("the side H is not compatible with the selected stride of {}".format(stride_h))
        print("img_h: {}, patch_h: {}, stride_h: {}".format(img_h, patch_h, stride_h))
        print("(img_h - patch_h) MOD stride_h: ", leftover_h)
        print("So the H dim will be padded with additional {} pixels ".format(stride_h - leftover_h))
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs

    # extend dimension of img w by adding zeros
    if leftover_w != 0:  # change dimension of img_w
        print("the side W is not compatible with the selected stride of {}".format(stride_w))
        print("img_w: {}, patch_w: {}, stride_w: {}".format(img_w, patch_w, stride_w))
        print("(img_w - patch_w) MOD stride_w: ", leftover_w)
        print("So the W dim will be padded with additional {} pixels ".format(stride_w - leftover_w))
        tmp_full_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape:", full_imgs.shape)
    return full_imgs


# data consinstency check
def data_consistency_check(imgs, masks):
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[2] == masks.shape[2])
    assert (imgs.shape[3] == masks.shape[3])
    assert (masks.shape[1] == 1)
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3)


# extract patches randomly in the full training images
def extract_random(full_imgs, full_gt, patch_h, patch_w, num_patches, channel):
    if num_patches % full_imgs.shape[0] != 0:
        print("N_subimags % Train_images_num should equal 0")
        sys.exit()

    # check the validity of input data
    assert (len(full_imgs.shape) == 4 and len(full_gt.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)  # check the channel is 1 or 3
    assert (full_gt.shape[1] == 1)  # gt has only one channel
    assert (full_imgs.shape[2] == full_gt.shape[2] and full_imgs.shape[3] == full_gt.shape[3])

    print(type(full_imgs))
    patches = np.empty((num_patches, full_imgs.shape[1], patch_h, patch_w), dtype=np.float16)
    patches_gt = np.empty((num_patches, full_gt.shape[1], patch_h, patch_w), dtype=np.uint8)
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image

    patch_per_img = num_patches // full_imgs.shape[0]  # N_patches equally divided in the full images
    print("patches per full image: {} ".format(patch_per_img))
    iter = 0  # iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        print("load image", i)
        k = 0
        while k < patch_per_img:
            x_center = random.randint(10 + int(patch_w / 2), img_w - int(patch_w / 2) - 10)
            y_center = random.randint(10 + int(patch_h / 2), img_h - int(patch_h / 2) - 10)
            # print "y_center " +str(y_center)
            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patch_gt = full_gt[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                       x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patches[iter] = patch
            patches_gt[iter] = patch_gt
            iter += 1  # total
            k += 1  # per full_img
    return patches, patches_gt


# split the full_imgs to pacthes
def extract_ordered(full_imgs, full_gt, patch_h, patch_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    N_patches_h = int(img_h / patch_h)  # round to lowest int

    if img_h % patch_h != 0:
        print("warning: {} patches in height, with about {} pixels left over"
              .format(N_patches_h, img_h % patch_h))

    N_patches_w = int(img_w / patch_w)  # round to lowest int

    if img_h % patch_h != 0:
        print("warning: {} patches in width, with about {} pixels left over"
              .format(N_patches_w, img_w % patch_w))
    print("number of patches per image: ", N_patches_h * N_patches_w)
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w), dtype=np.float16)
    patches_gt = np.empty((N_patches_tot, full_gt.shape[1], patch_h, patch_w), dtype=np.uint8)
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        print(i)
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, :, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w]
                patch_gt = full_gt[i, :, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w]
                patches[iter_tot] = patch
                patches_gt[iter_tot] = patch_gt
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches, patches_gt  # array with all the full_imgs divided in patches


# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)

    num_patches_one = ((img_h - patch_h) // stride_h + 1) \
                      * ((img_w - patch_w) // stride_w + 1)

    num_patches_total = num_patches_one * full_imgs.shape[0]
    print("Number of patches on h : ", (img_h - patch_h) // stride_h + 1)
    print("Number of patches on w : ", (img_w - patch_w) // stride_w + 1)
    print("number of patches per image: {}, totally for this testing dataset: {}"
          .format(num_patches_one, num_patches_total))
    patches = np.empty((num_patches_total, full_imgs.shape[1], patch_h, patch_w), dtype=np.float16)
    iter_total = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_total] = patch
                iter_total += 1  # total
    assert (iter_total == num_patches_total)
    return patches  # array with all the full_imgs divided in patches


def recompose_overlap(preds, img_h, img_w, stride_h, stride_w, channel, loss_weight=None):
    assert (len(preds.shape) == 4)  # 4D arrays
    #assert (preds.shape[1] == channel)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]

    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w

    print("N_patches_h: ", N_patches_h)
    print("N_patches_w: ", N_patches_w)
    print("N_patches_img: ", N_patches_img)
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img

    print("According to the dimension inserted, there are {} full images (of {} x {} each)"
          .format(N_full_imgs, img_h, img_w))

    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # initialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches

    # extract each patch
    center = [patch_h / 2, patch_w / 2]
    expand = patch_h / 2
    left = center[1] - expand
    right = center[1] + expand
    top = center[0] - expand
    bottom = center[0] + expand

    if loss_weight is not None:
        weight = np.reshape(loss_weight, (patch_h, patch_w))
	weight += 0.000000001
    else:
        weight = 1

    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h + top:(h * stride_h) + bottom,
                          w * stride_w + left:(w * stride_w) + right] += \
                    preds[k, :, top:bottom, left:right]*weight
                full_sum[i, :, h * stride_h + top:(h * stride_h) + bottom,
                         w * stride_w + left:(w * stride_w) + right] += weight
                k += 1
    #assert (k == preds.shape[0])
    #assert (np.min(full_sum) >= 0.0)  # must larger than 0
    #print(np.min(full_sum))

    final_avg = full_prob / (full_sum + 0.0000000001)
    #print("the shape of prediction result", final_avg.shape)
    #print("max value of prediction result", np.max(final_avg))
    #assert (np.max(final_avg) <= 1.01)  # max value for a pixel is 1.0
    #assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
    return final_avg


# Recompone the full images with the patches
def recompose(data, N_h, N_w, channel):
    assert (data.shape[1] == channel)  # check the channel is 1 or 3
    assert (len(data.shape) == 4)
    N_patch_per_img = N_w * N_h
    assert (data.shape[0] % N_patch_per_img == 0)
    N_full_imgs = data.shape[0] / N_patch_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]

    # define and start full recompone
    full_recomp = np.empty((N_full_imgs, data.shape[1], N_h * patch_h, N_w * patch_w))
    k = 0  # iter full img
    s = 0  # iter single patch
    while s < data.shape[0]:
        # recompone one:
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


# Extend the full images becasue patch divison is not exact
def paint_border(data, patch_h, patch_w, channel):
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == channel)  # check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]

    if (img_h % patch_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = (img_h / patch_h + 1) * patch_h
    if (img_w % patch_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = (img_w / patch_w + 1) * patch_w

    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:, :, 0:img_h, 0:img_w] = data[:, :, :, :]
    return new_data
