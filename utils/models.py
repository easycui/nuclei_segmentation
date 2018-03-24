from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.activations import selu, relu
from keras.layers.core import Lambda
import tensorflow as tf


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            print(len(outputs))
            if gpu_count > 1:
                merged.append(layers.concatenate(outputs, axis=0))
            else:
                merged.append(outputs[0])

        return Model(input=model.inputs, output=merged)

def unet(n_ch, patch_height, patch_width, category_num, act='selu', loss_weight=None, sample_weight_mode=None, GPU_num=1, net_name='unet16', fine_tune = 1, pretrain_model = ''):
    nets = {'unet': unet_backbone,
            
            }
    learn_rates = {'unet': 1e-4,
                  
                   }
    net = nets[net_name]
    learn_rate = learn_rates[net_name]
    inputs = Input((n_ch, patch_height, patch_width))
    output = net(inputs, act)
    conv = Conv2D(category_num, (1, 1), activation=act, padding='same', data_format='channels_first')(output)
    conv = core.Reshape((category_num, patch_height * patch_width))(conv)
    conv = core.Permute((2, 1))(conv)
    ############
    conv = core.Activation('softmax')(conv)
    model = Model(inputs=inputs, outputs=conv)
    if fine_tune == 1:
        model.load_weights(pretrain_model, by_name = True)
        for layer in model.layers[10:]:
            layer.trainable = False
    if GPU_num > 1:
        model = make_parallel(model, GPU_num)
    adam = Adam(lr=learn_rate)
    model.compile(optimizer=adam,
          loss='categorical_crossentropy',
          metrics=['accuracy'],
          loss_weights=loss_weight,
          sample_weight_mode=sample_weight_mode)
    return model


def unet_backbone(inputs, act):

    conv1 = Conv2D(32, (3, 3), activation=act, padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation=act, padding='same', data_format='channels_first')(conv1)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, (3, 3), activation=act, padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation=act, padding='same', data_format='channels_first')(conv2)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same', data_format='channels_first')(conv3)
    conv3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(256, (3, 3), activation=act, padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation=act, padding='same', data_format='channels_first')(conv4)
    conv4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(512, (3, 3), activation=act, padding='same', data_format='channels_first')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(256, (3, 3), activation=act, padding='same', data_format='channels_first')(conv5)
    conv5 = Dropout(0.2)(conv5)
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)
    up1 = layers.concatenate([up1, conv4], axis=1)

    conv6 = Conv2D(256, (3, 3), activation=act, padding='same', data_format='channels_first')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation=act, padding='same', data_format='channels_first')(conv6)
    conv6 = Dropout(0.2)(conv6)
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up2 = layers.concatenate([up2, conv3], axis=1)

    conv7 = Conv2D(128, (3, 3), activation=act, padding='same', data_format='channels_first')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation=act, padding='same', data_format='channels_first')(conv7)
    conv7 = Dropout(0.2)(conv7)
    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = layers.concatenate([up3, conv2], axis=1)

    conv8 = Conv2D(64, (3, 3), activation=act, padding='same', data_format='channels_first')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(32, (3, 3), activation=act, padding='same', data_format='channels_first')(conv8)
    conv8 = Dropout(0.2)(conv8)
    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = layers.concatenate([up4, conv1], axis=1)

    #
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same', data_format='channels_first')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same', data_format='channels_first')(conv9)
    conv9 = Dropout(0.2)(conv9)
    return conv9

