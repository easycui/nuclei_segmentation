# ==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#  Read the raw images and store them in hdf5 files
# ============================================================
from __future__ import print_function
import os
import h5py
import sys
import numpy as np
from PIL import Image
import utils.help_functions as hf

if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print('The config file does not exist')
        sys.exit()
else:
    config_file = './configuration.txt'
config = hf.parse_config(config_file)


def prepare_training_dataset():

    # hdf5 files
    train_images_file = config['train_images_file']
    train_groundTruth_file = config['train_gt_file']
    train_masks_file = config['train_masks_file']

    # image properties
    channels = config['channels']
    height = config['train_height']
    width = config['train_width']
    category = config['category']
    # ------------Path of the images --------------------------------------------------------------
    # train dataset path
    N_train_images = config['n_train_images']
    imgs_train_dir = config['imgs_train_dir']
    groundTruth_train_dir = config['gt_train_dir']
    masks_train_dir = config['mask_train_dir']

    # ---------------------------------------------------------------------------------------------
    # the path to write hdfs files
    dataset_path = config['dataset_path']

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if N_train_images == 0:
        return
    # getting the training datasets
    imgs_train, groundTruth_train, masks_train = hf.get_dataset(imgs_train_dir,
                                                                groundTruth_train_dir,
                                                                masks_train_dir,
                                                                N_train_images,
                                                                height,
                                                                width,
                                                                channels,
                                                                category)
    print("saving train datasets to", dataset_path)
    # all the data are float64
    hf.write_hdf5(imgs_train, train_images_file)
    hf.write_hdf5(groundTruth_train, train_groundTruth_file)
    hf.write_hdf5(masks_train, train_masks_file)


def prepare_test_dataset():


    #print config
    # hdf5 files
    test_images_file = config['test_images_file']
    test_groundTruth_file = config['test_gt_file']
    test_masks_file = config['test_masks_file']

    # image properties
    channels = config['channels']


    height = config['test_height']
    width = config['test_width']
    category = config['category']

    # ------------Path of the images --------------------------------------------------------------
    N_test_images = config['n_test_images']
    imgs_test_dir = config['imgs_test_dir']
    groundTruth_test_dir = config['gt_test_dir']
    masks_test_dir = config['mask_test_dir']
    # ---------------------------------------------------------------------------------------------
    dataset_path = config['dataset_path']
    if N_test_images == 0:
        return
    if not os.path.exists(dataset_path):
        print("traning image doesn't exist")
        exit(0)
    if not os.path.exists(imgs_test_dir):
        print("test image doesn't exist")
        exit(0)
    # getting the testing datasets
    imgs_test, groundTruth_test, masks_test = hf.get_dataset(imgs_test_dir,
                                                             groundTruth_test_dir,
                                                             masks_test_dir,
                                                             N_test_images,
                                                             height,
                                                             width,
                                                             channels,
                                                             category)
    print("saving test datasets to" + dataset_path)
    hf.write_hdf5(imgs_test, test_images_file)
    hf.write_hdf5(groundTruth_test, test_groundTruth_file)
    hf.write_hdf5(masks_test, test_masks_file)

prepare_training_dataset()
prepare_test_dataset()
