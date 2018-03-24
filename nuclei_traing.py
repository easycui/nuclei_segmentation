###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import SGD
import sys
import os
import random
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale
from skimage import io
import matplotlib.pyplot as plt
sys.path.append('./')
import utils.help_functions as hf
import utils.extract_patches as ep
import utils.models as md
from scipy import ndimage
from skimage.transform import rotate

def main(config_file):

    # set the memory usage
    np.set_printoptions(threshold=np.nan)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))
    # ========= Load settings from Config file
    config = hf.parse_config(config_file)
    name_experiment = config['name']
    fine_tuning = config['fine_tuning']
    experiment_path = './experiment/' + name_experiment
    print("--------------Generating the patches-----------------------")
    # ============ Load the data and divided in patches
    # patches_imgs_train is float16 in range 0.0-255.0, patches_gt_train is
    patches_imgs_train, patches_gt_train, category_num = ep.get_data_training(
        train_imgs_file=config['train_images_file'],
        train_gt_file=config['train_gt_file'],
        patch_height=config['patch_height'],
        patch_width=config['patch_width'],
        num_patches=config['n_patches'],
        channel=config['channels'],
        config = config,
    )
    assert config['category'] == category_num
    print("extract patches done!")

    print("--------------Generating the sample data-------------------")
    # ========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0], 40)
    hf.visualize(hf.group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
                 experiment_path + "/sample_input_imgs")

    sample_imgs = hf.label2rgb(patches_gt_train[0:N_sample, :, :, :], category_num)
    hf.visualize(hf.group_images(sample_imgs, 5), experiment_path + "/sample_input_gt")

    print("-----------------transform gt to one hot target------------------")
    patches_gt_train = hf.get_gt(patches_gt_train, category_num)
    print("one hot target shape:", patches_gt_train.shape)
    print("one hot target datatype:", patches_gt_train.dtype)


    print("--------------Construct the network------------------------")
    # =========== Construct and save the model arcitecture =====
    n_ch = patches_imgs_train.shape[1]
    patch_height = patches_imgs_train.shape[2]
    patch_width = patches_imgs_train.shape[3]
    use_weighted = config['loss_weight']
    batch_size = config['batch_size']
    border = config['border']
    loss_weight = hf.get_loss_weight(config['patch_height'], config['patch_width'], use_weighted, border=border)
    if loss_weight is not None:
        loss_weight = [loss_weight]
    #print(loss_weight)
    #select the model
    net = config['net']
    use_sample_weights = config['use_sample_weight']
    positive_weight = config['positive_weight']
    GPU_num = config['gpu_num']

    if use_sample_weights == 1:
        sample_weight_mode = 'temporal'
    else:
        sample_weight_mode = None

    pretrain_model = experiment_path + '/pretrain_weight.h5'
    if fine_tuning == 1:
        if os.path.isfile(pretrain_model):
            print("----------load pretrained model---------------")
            model = md.unet(n_ch, patch_height, patch_width, category_num, 'selu',
                            loss_weight, sample_weight_mode, GPU_num, net, 1, pretrain_model)
        else:
            print('pretrained model cannot find')
            sys.exit()
    else:
        model = md.unet(n_ch, patch_height, patch_width, category_num, 'selu',
                        loss_weight, sample_weight_mode, GPU_num, net, 0)
    print(model.summary())



    print("Check: final output of the network:", model.output_shape)
    print('model.metrics_names:', model.metrics_names)
    #plot_model(model, to_file=experiment_path + '/' + net + '_model.png')  # check how the model looks like
    json_string = model.to_json()
    open(experiment_path + '/' + net + '_architecture.json', 'w').write(json_string)

    # ============  Training ==================================
    # save at each epoch if the validation decreased
    N_epoch = config['n_epochs']

    weight_path = experiment_path + '/' + net + '_best_weights.h5'
    checkpointer = ModelCheckpoint(filepath=weight_path,
                                   verbose=1,
                                   monitor='val_loss',
                                   mode='auto',
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir=experiment_path + '/logs', batch_size=batch_size)

    # def step_decay(epoch):
    #     lrate = 0.01 #the initial learning rate (by default in keras)
    #     if epoch==100:
    #         return 0.005
    #     else:
    #         return lrate
    #
    # lrate_drop = LearningRateScheduler(step_decay)
    # transform gt to target

    class _LossHistory(Callback):
        def __init__(self):
            self.best = np.inf
        def on_epoch_end(self, epoch, logs=None):
            #print("--------------------", logs.get('loss'))
            #print("--------------------", logs.get('val_loss').shape)
            self.loss = np.mean(logs.get('loss'))
            self.val_loss = np.mean(logs.get('val_loss'))
            if epoch % 10 == 0:
                filename = "{}/{}_{}_weight.h5".format(experiment_path, net, epoch)
                self.model.save_weights(filename, overwrite=True)
                #Client.messages.create(body = message, from_=my_twilio_number, to=dest_cellphone)
            if self.val_loss < self.best:
                filename = "{}/{}_best_weight.h5".format(experiment_path, net)
                self.model.save_weights(filename,overwrite=True)
                self.best = self.val_loss
            self.model.save_weights(experiment_path + '/' + net + '_last_weight.h5',overwrite=True)

    history = _LossHistory()
    print('============================================================')
    print("The training ", name_experiment)
    print("starts at:", time.strftime('%X %x %Z'))
    print('------------------------------------------------------------')



    def elastic_transform(img, gt, alpha, sigma, alpha_affine, random_state=None):
        # the img and gt must be in uint8 data type
        if random_state is None:
            random_state = np.random.RandomState(None)
        img_shape = img.shape
        gt_shape = gt.shape
        shape_size = img_shape[:2]
        #print(shape)
        #print(img)
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)

        img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)
        img = np.reshape(img, img_shape)
        gt = cv2.warpAffine(gt, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)
        ax = random_state.rand(*shape_size) * 2 - 1
        ay = random_state.rand(*shape_size) * 2 - 1
        max_channel = max(img_shape[2], gt_shape[2])
        ax_all = np.zeros((img_shape[0], img_shape[1], max_channel))
        ay_all = np.zeros((img_shape[0], img_shape[1], max_channel))
        #print(img_shape, gt_shape)
        for i in range(max_channel):
            ax_all[:,:,i] = ax
            ay_all[:,:,i] = ay
        ax_img = ax_all[:,:,:img_shape[2]]
        ay_img = ay_all[:,:,:img_shape[2]]
        dx = gaussian_filter(ax_img, sigma) * alpha
        dy = gaussian_filter(ay_img, sigma) * alpha
        x, y, z = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]), np.arange(img_shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
        img = map_coordinates(img, indices, order=1, mode='constant').reshape(img_shape)


        #gt = cv2.warpAffine(gt, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        ax_gt = ax_all[:,:,:gt_shape[2]]
        ay_gt = ay_all[:,:,:gt_shape[2]]

        dx = gaussian_filter(ax_gt, sigma) * alpha
        dy = gaussian_filter(ay_gt, sigma) * alpha
        x, y, z= np.meshgrid(np.arange(gt_shape[1]), np.arange(gt_shape[0]), np.arange(gt_shape[2]))
        indices_gt = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
        gt = map_coordinates(gt, indices_gt, order=1, mode='constant').reshape(gt_shape)

        return img, gt


    def rescale_transform(img, gt):
        scale = random.random() + 0.5  # range in 0.5-1.5
        #print(scale)
        #scale = random.random() * 1.75 + 0.25  # range in 0.25-2
        size = img.shape[0]
        if scale >= 1:
            img_scaled = rescale(img, scale)
            gt_scaled = rescale(gt, scale)
            img_scaled = img_scaled[:size,:size, :]
            gt_scaled = gt_scaled[:size, :size, :]
        else:
            img_scaled_1 = rescale(img, scale)
            gt_scaled_1 = rescale(gt, scale)
            #print(img_scaled_1.dtype)
            img_scaled = np.zeros(img.shape)
            gt_scaled = np.zeros(gt.shape)
            offset = (img_scaled.shape[0] - img_scaled_1.shape[0])/2
            #print(offset)
            img_scaled[offset:offset+img_scaled_1.shape[0], offset:offset+img_scaled_1.shape[0], :]=img_scaled_1
            gt_scaled[offset:offset+img_scaled_1.shape[0], offset:offset+img_scaled_1.shape[0], :]=gt_scaled_1
        img_scaled *= 255
        gt_scaled *= 255
        img_scaled = img_scaled.astype(np.uint8)
        gt_scaled = gt_scaled.astype(np.uint8)
        return img_scaled, gt_scaled


    def deformation_generator(imags_train, gt_train, batch_size=32, use_sample_weights=0, positive_weight=1, use_rescale = 0, channel = 2):
        while True:
            select = random.randint(0, imags_train.shape[0] - batch_size - 1)
            img_patches = np.copy(imgs_train[select:select+batch_size, :, :, :])
            img_patches = np.transpose(img_patches, (0, 2, 3, 1))
            img_patches = img_patches.astype(np.uint8)
            #orig = np.copy(img_patches)
            gt_patches = np.copy(gt_train[select:select+batch_size, :, :])

            gt_patches = np.reshape(gt_patches, (batch_size, patch_height, patch_width, channel))
            gt_patches = (gt_patches * 255).astype(np.uint8)
            #gt_orig = np.copy(gt_patches)
            # the input of elastic_transform must be 0-255 in uint8 type
            #print(img_patches[0,:,:,:])

            #rotate
            if config['rotate'] == 1:
                for i in range(batch_size):
                    angle = int(random.random() * 360)
                    #print(angle)
                    img_patches[i,:,:,:] = (rotate(img_patches[i,:,:,:], angle, order=0)*255).astype(np.uint8)
                    gt_patches[i,:,:,:] = (rotate(gt_patches[i,:,:,:], angle,order=0)*255).astype(np.uint8)
            #print(img_patches[0,:,0,0])
            #orig = np.copy(img_patches)
            #gt_orig = np.copy(gt_patches)
            #flip
            if config['flip'] == 1:
                for i in range(batch_size):
                    rand_decision = int(random.random() + 0.5)
                    if rand_decision == 1:
                        img_patches[i,:,:,:] = img_patches[i,::-1,:,:]
                        gt_patches[i,:,:,:] = gt_patches[i,::-1,:,:]
                    rand_decision = int(random.random() + 0.5)
                    if rand_decision == 1:
                        img_patches[i,:,:,:] = img_patches[i,:,::-1,:]
                        gt_patches[i,:,:,:] = gt_patches[i,:,::-1,:]
            #shift
            if config['shift'] == 1:
                img_patches_tmp = np.zeros(img_patches.shape, dtype=np.uint8)
                gt_patches_tmp = np.zeros(gt_patches.shape, dtype=np.uint8)
                height = img_patches.shape[1]
                width = img_patches.shape[2]
                for i in range(batch_size):

                    v_shift = int(random.random() * height/2) - int(height/4)
                    h_shift = int(random.random() * width/2) - int(width/4)
                    if v_shift == 0 or h_shift == 0:
                        img_patches_tmp[i,:,:,:] = img_patches[i,:,:,:]
                        gt_patches_tmp[i,:,:,:] = gt_patches[i,:,:,:]
                    elif v_shift > 0 and h_shift > 0:
                        img_patches_tmp[i,:-v_shift,:-h_shift,:] = img_patches[i, v_shift: ,h_shift:,:]
                        gt_patches_tmp[i, :-v_shift, :-h_shift, :] = gt_patches[i, v_shift:, h_shift:, :]
                    elif v_shift < 0 and h_shift > 0:
                        img_patches_tmp[i, -v_shift:, :-h_shift,:] = img_patches[i, :v_shift, h_shift:, :]
                        gt_patches_tmp[i, -v_shift:, :-h_shift, :] = gt_patches[i, :v_shift, h_shift:, :]
                    elif h_shift < 0 and v_shift > 0 :
                        img_patches_tmp[i, :-v_shift, -h_shift:, :] = img_patches[i, v_shift:, :h_shift, :]
                        gt_patches_tmp[i,:-v_shift ,-h_shift:,:] = gt_patches[i, v_shift:, :h_shift, :]
                    elif v_shift < 0 and h_shift < 0:
                        img_patches_tmp[i, -v_shift:, -h_shift:,:] = img_patches[i, :v_shift, :h_shift, :]
                        gt_patches_tmp[i, -v_shift:, -h_shift:, :] = gt_patches[i, :v_shift, :h_shift, :]
                img_patches = img_patches_tmp
                gt_patches = gt_patches_tmp
            #print('----------------', config['deformation'])
            if config['deformation'] == 1:
                for i in range(batch_size):
                    alpha = random.randint(100,200)
                    #sigma = random.randint(8,15)
                    sigma = 12
                    #print('*****************************')
                    img_patches[i, :, :, :], gt_patches[i, :, :, :] = \
                        elastic_transform(img_patches[i, :, :, :], gt_patches[i, :, :, :], alpha, sigma, 10)
            if use_rescale == 1:
                for i in range(batch_size):
                    img_patches[i,:,:,:], gt_patches[i,:,:,:] = \
                        rescale_transform(img_patches[i, :, :, :], gt_patches[i, :, :, :])
            #blur

            if config['blur'] == 1:
                for i in range(batch_size):
                    if random.random() > 0.5:
                        Kernel_size = int(random.random() * 10 + 1)
                        kernel = np.ones((Kernel_size, Kernel_size), np.float32) / Kernel_size ** 2
                        img_patches[i,:,:,:] = cv2.filter2D(img_patches[i,:,:,:], -1, kernel)
            #orig_gt_patches = np.copy(gt_patches)

            edge = gt_patches[:,:,:,1]
            edge[np.where(edge>255*0.5)] = 255
            edge[np.where(edge<255)] = 0

            cell = gt_patches[:,:,:,2]
            cell[np.where(cell>0)] = 255
            cell[np.where(edge==255)] = 0

            negative = gt_patches[:,:,:,0]
            #negative[np.where(edge==0)] = 255
            #negative[np.where(cell==0)] = 255
            negative[:,:,:] = 255
            negative[np.where(edge>0)] = 0
            negative[np.where(cell>0)] = 0

            distance = np.copy(gt_patches[:,:,:,1])
            gt_patches = np.reshape(gt_patches, (batch_size, patch_width * patch_height, channel))
            gt_patches = gt_patches.astype(np.float32)
            gt_patches = gt_patches/255.0
            #
            img_patches = np.transpose(img_patches, (0, 3, 1, 2))
            img_patches = img_patches.astype(np.float32)
            img_patches /= 255.0
            if use_sample_weights == 1:
                sample_weights = np.ones(distance.shape, dtype=np.float16)
                for i in range(batch_size):
                    distance[i,:,:] = distance_transform_edt(distance[i,:,:])
                    sample_weights[i,:,:] += distance[i,:,:]/4
                sample_weights[np.where(distance == 0)] = 2
                #sample_weights[np.where(gt_patches[:,:,1]==1)] = positive_weight
                sample_weights = np.reshape(sample_weights, gt_patches.shape[0:2])
                yield (img_patches, gt_patches, sample_weights)
            else:
                yield (img_patches, gt_patches)

    def validation_generator(imags_vali, gt_vali, batch_size=32):
        while True:
            select = random.randint(0, imags_vali.shape[0] - batch_size - 1)
            img_patches = imgs_vali[select:select+batch_size, :, :, :]
            gt_patches = gt_vali[select:select+batch_size, :, :]
            yield img_patches, gt_patches
    augment = config['augment']
    use_rescale = config['use_rescale']
    channel = config['channels']
    if augment == 1:
        if config['shuffle'] == 1:
            index = np.arange(patches_imgs_train.shape[0])
            np.random.shuffle(index)
            #print(index)
            patches_imgs_train = patches_imgs_train[index, :, :, :]
            patches_gt_train = patches_gt_train[index, :, :]
        split_factor = config['split']/10.0
        num_patches = patches_imgs_train.shape[0]
        train_num = int(split_factor*num_patches)

        imgs_train = patches_imgs_train[0:train_num, :, :, :]
        gt_train = patches_gt_train[0:train_num, :, :]

        imgs_vali = patches_imgs_train[train_num:, :, :, :]
        imgs_vali = np.transpose(imgs_vali, (0, 2, 3, 1))
        imgs_vali = imgs_vali.astype(np.uint8)
        if config['blur'] == 1:
            for i in range(imgs_vali.shape[0]):
                Kernel_size = int(random.random() * 10 + 1)
                kernel = np.ones((Kernel_size, Kernel_size), np.float32) / Kernel_size ** 2
                #print(imgs_vali[i,:,:,:].shape)
                imgs_vali[i, :, :, :] = cv2.filter2D(imgs_vali[i, :, :, :], -1, kernel)
        imgs_vali = np.transpose(imgs_vali, (0, 3, 1, 2))
        imgs_vali = imgs_vali/255.0
        gt_vali = patches_gt_train[train_num:, :, :]
        #
        #generator = deformation_generator(imgs_train, gt_train, batch_size, use_sample_weights, positive_weight, use_rescale, channel)
        #a, b ,c, d = generator.next()
        #plt.figure()
        #io.imshow(a[0,:,:,:])
        #plt.figure()
        #io.imshow(b[0,:,:,:])
        #plt.figure()
        #io.imshow(c[0,:,:,:])
        #plt.figure()
        #io.imshow(d[0,:,:,:])
        #plt.show()
        #time.sleep(5000)

        class_weight = None
        print(imgs_train.shape[0])
        print(imgs_vali.shape[0])
        model.fit_generator(deformation_generator(imgs_train, gt_train, batch_size, use_sample_weights, positive_weight, use_rescale, channel), epochs=N_epoch,
                            verbose=2, shuffle=True, steps_per_epoch=imgs_train.shape[0]//batch_size,
                            validation_data=(imgs_vali, gt_vali), callbacks=[history],
                            workers=4, use_multiprocessing=True, class_weight=class_weight)
    else:
        #patches_imgs_train = patches_imgs_train.astype(np.float32)
        patches_imgs_train /= 255.
        model.fit(patches_imgs_train, patches_gt_train, epochs=N_epoch,
                  batch_size=batch_size, verbose=2, shuffle=True,
                  validation_split=0.2, callbacks=[history])
    # ========== Save and test the last model ===================
    model.save_weights(experiment_path + '/' + net + '_last_weights.h5', overwrite=True)


if __name__ == '__main__':
    main(sys.argv[1])
