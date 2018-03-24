###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

# Python
from __future__ import print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
# Keras
from keras.models import model_from_json
from keras.models import Model
import tensorflow as tf
# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
import os
import shutil
sys.path.insert(0, './')
# help_functions.py
import utils.help_functions as hf
# extract_patches.py
import utils.extract_patches as ep
# pre_processing.py

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = './configuration.txt'
# ========= CONFIG FILE TO READ FROM =======
config = hf.parse_config(config_file)

# ===========================================
# run the training on invariant or local
dataset = config['dataset']
# original test images

test_images_file = config['test_images_file']
test_images = hf.load_hdf5(test_images_file)
full_img_height = test_images.shape[2]
full_img_width = test_images.shape[3]

# the masks
test_masks_file = config['test_masks_file']
test_border_masks = hf.load_hdf5(test_masks_file)
test_gt_file = config['test_gt_file']

# dimension of the patches
patch_height = config['patch_height']
patch_width = config['patch_width']
# the stride in case output with average
stride_height = config['stride_height']
stride_width = config['stride_width']

# make sure there is no gap between patches
assert (stride_height <= patch_height and stride_width <= patch_width)

# model name
name_experiment = config['name']
path_experiment = './experiment/' + name_experiment + '/'
batch_size = config['batch_size']
# N full images to be predicted
n_test_images = config['n_test_images']
test_result = str(config['test_result_path'])

average_mode = config['average_mode']
net = config['net']
real_value = config['real_value']

# ============ Load the data and divide in patches
extended_height = 0
extended_width = 0
test_gt = None
category_num = config['category']
channel = config['channels']
use_weight = config['loss_weight']

test_result = path_experiment + test_result + '/'
if os.path.exists(test_result):
    print('test dir existing, continue?')
    answer = raw_input()
    if answer == 'y' or answer == 'yes':
        shutil.rmtree(test_result)
        pass
    else:
        sys.exit()

os.mkdir(test_result)
os.system('cp ' + config_file + ' ' + test_result)
if average_mode:
    patches_imgs_test, extended_height, extended_width, test_gt, category = ep.get_data_testing_overlap(
        test_images_file=test_images_file,  # original
        test_gt_file=test_gt_file,  # ground truth
        n_test_images=n_test_images,
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width,
        channel=channel,
        config = config
    )

else:
    patches_imgs_test, extended_height, extended_width, test_gt, category = ep.get_data_testing(
        test_images_file=test_images_file,  # original
        test_gt_file=test_gt_file,  # ground truth
        n_test_images=n_test_images,
        patch_height=patch_height,
        patch_width=patch_width,
        channel=channel,
        config = config
    )

patches_imgs_test /= 255.0
# ================ Run the prediction of the patches ==================================
best_last = config['best_last']
# Load the saved model
model = model_from_json(open(path_experiment + net + '_architecture.json').read(), custom_objects={'tf': tf})
print(model.summary())

if best_last == 1:
    weight_file = path_experiment + net + '_best_weight.h5'
elif best_last == -1:
    weight_file = path_experiment + net + '_last_weight.h5'
else:
    weight_file = path_experiment + net + '_{}_weight.h5'.format(best_last)


if os.path.isfile(weight_file):
    model.load_weights(weight_file)
else:
    print("the model is not found")
    sys.exit()

shape = patches_imgs_test.shape
num_patches = shape[0]/batch_size + 1
expand_patches_imgs_test = np.zeros((num_patches * batch_size - shape[0], shape[1], shape[2], shape[3]), dtype=np.float16)
patches_imgs_test = np.concatenate((patches_imgs_test, expand_patches_imgs_test), axis = 0)
# Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=batch_size, verbose=2)
predictions = predictions[:shape[0],:,:]
print("predicted patches size :", predictions.shape)
print("max value of the predicted patches {}".format(np.max(predictions[:,:,1])))
# ===== Convert the prediction arrays in corresponding images
pred_patches = hf.pred_to_imgs(predictions, real_value=real_value)


# ========== Elaborate and visualize the predicted images ====================
print("---------------recompose wholes from patches---------------------")
border = config['border']
loss_weight = hf.get_loss_weight(patch_height, patch_width, use_weight, border=border)
if average_mode:
    pred_imgs = ep.recompose_overlap(pred_patches, extended_height,
                                     extended_width, stride_height, stride_width,
                                     channel, loss_weight)  # predictions
else:
    pred_imgs = ep.recompose(pred_patches, extended_height, extended_width)  # predictions

gt_imgs = test_gt
# back to original dimensions
orig_imgs = test_images[0:pred_imgs.shape[0], :, :, :]  # originals
pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
print("max value of final prediction reulst", np.max(pred_imgs))
gt_imgs = gt_imgs[:, :, 0:full_img_height, 0:full_img_width]

#print(pred_imgs)
print("Orig imgs shape: ", orig_imgs.shape)
print("pred imgs shape: ", pred_imgs.shape)
print("Gtruth imgs shape: ", gt_imgs.shape)

gt_imgs = hf.label2rgb(gt_imgs, category_num)

for i in range(orig_imgs.shape[0]):
    hf.visualize(np.transpose(orig_imgs[i,:,:,:], (1,2,0)), test_result + str(i) + "_originals")
    hf.visualize(np.transpose(pred_imgs[i,:,:,:], (1,2,0)), test_result + str(i) + "_predictions")
    #hf.visualize(np.transpose(pred_patches[i, :, :, :], (1, 2, 0)), test_result + str(i) + "_predictions")
    hf.visualize(np.transpose(gt_imgs[i,:,:,:], (1,2,0)), test_result + str(i) + "_groundTruths")
#
# for i in range(orig_imgs.shape[0]):
#     img = orig_imgs[i, 1, :, :]
#     gt = gt_imgs[i, 0, :, :]
#     img[np.where(gt>0)] = 255
#     img = orig_imgs[i, 2, :, :]
#     pred = pred_imgs[i, 0, :, :]
#     img[np.where(pred>0.3)] = 255
#
# for i in range(orig_imgs.shape[0]):
#     hf.visualize(np.transpose(orig_imgs[i,:,:,:], (1,2,0)), test_result + str(i) + '_gt_marker')
# # hf.visualize(hf.group_images(orig_imgs, path_experiment + str(i) + "_originals")
# # hf.visualize(hf.group_images(pred_imgs, N_visual), path_experiment + str(i) + "_predictions")
# #
# # hf.visualize(hf.group_images(gt_imgs, N_visual), path_experiment + str(i) + "_groundTruths")
#
#
# # ====== Evaluate the results
# print("========  Evaluate the results =======================")
# gt_imgs = gt_imgs[:,0,:,:]/255
# gt_imgs = gt_imgs[:,94:290, 94:290]
# pred_imgs = pred_imgs[:, 0,94:290,94:290]
# error = np.square(pred_imgs - gt_imgs)
# print(error)
# error = np.mean(error)
# print("---------------------------------------------")
# print('error is: {}'.format(error))
# print("---------------------------------------------")
#
# gt_imgs = np.reshape(gt_imgs, gt_imgs.size)
# pred_imgs = np.reshape(pred_imgs, pred_imgs.size)
# fpr, tpr, thresholds = roc_curve(gt_imgs, pred_imgs, pos_label=1)
# AUC_ROC = roc_auc_score(gt_imgs, pred_imgs)
# # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
# print("Area under the ROC curve: ", AUC_ROC)
# roc_curve = plt.figure()
# plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = {:.4f} error = {:.4f}'.format(AUC_ROC, error))
# plt.title('ROC curve')
# plt.xlabel("FPR (False Positive Rate)")
# plt.ylabel("TPR (True Positive Rate)")
# plt.legend(loc="lower right")
# plt.savefig(test_result + "ROC.png")


# # Area under the ROC curve
# fpr, tpr, thresholds = roc_curve((y_true), y_scores)
# AUC_ROC = roc_auc_score(y_true, y_scores)
# # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
# print "\nArea under the ROC curve: " + str(AUC_ROC)
# roc_curve = plt.figure()
# plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
# plt.title('ROC curve')
# plt.xlabel("FPR (False Positive Rate)")
# plt.ylabel("TPR (True Positive Rate)")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment + "ROC.png")
#
# # Precision-recall curve
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
# recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
# AUC_prec_rec = np.trapz(precision, recall)
# print "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
# prec_rec_curve = plt.figure()
# plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
# plt.title('Precision - Recall curve')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment + "Precision_recall.png")
#
# # Confusion matrix
# threshold_confusion = 0.5
# print "\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion)
# y_pred = np.empty((y_scores.shape[0]))
# for i in range(y_scores.shape[0]):
#     if y_scores[i] >= threshold_confusion:
#         y_pred[i] = 1
#     else:
#         y_pred[i] = 0
# confusion = confusion_matrix(y_true, y_pred)
# print confusion
# accuracy = 0
# if float(np.sum(confusion)) != 0:
#     accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
# print "Global Accuracy: " + str(accuracy)
# specificity = 0
# if float(confusion[0, 0] + confusion[0, 1]) != 0:
#     specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
# print "Specificity: " + str(specificity)
# sensitivity = 0
# if float(confusion[1, 1] + confusion[1, 0]) != 0:
#     sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
# print "Sensitivity: " + str(sensitivity)
# precision = 0
# if float(confusion[1, 1] + confusion[0, 1]) != 0:
#     precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
# print "Precision: " + str(precision)
#
# # Jaccard similarity index
# jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
# print "\nJaccard similarity score: " + str(jaccard_index)
#
# # F1 score
# F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
# print "\nF1 score (F-measure): " + str(F1_score)
#
# # Save the results
# file_perf = open(path_experiment + 'performances.txt', 'w')
# file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
#                 + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
#                 + "\nJaccard similarity score: " + str(jaccard_index)
#                 + "\nF1 score (F-measure): " + str(F1_score)
#                 + "\n\nConfusion matrix:"
#                 + str(confusion)
#                 + "\nACCURACY: " + str(accuracy)
#                 + "\nSENSITIVITY: " + str(sensitivity)
#                 + "\nSPECIFICITY: " + str(specificity)
#                 + "\nPRECISION: " + str(precision)
#                 )
# file_perf.close()

