###################################################
#
#   Script to launch the training
#
##################################################
from __future__ import print_function
import os, sys
from utils.help_functions import parse_config
import shutil

# config file to read from
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print('The config file does not exist')
        sys.exit()
else:
    config_file = './configuration.txt'
config = parse_config(config_file)
# ===========================================
# name of the experiment
name_experiment = config['name']
dataset = config['dataset']
nohup = config['train_nohup']  # std output on log file?
fine_tuning = config['fine_tuning']
pretrained_model = config['pretrain_model']
run_GPU = 'THEANO_FLAGS=device=gpu,floatX=float32 '
net = config['net']

# create a folder for the results
result_dir = './experiment/' + name_experiment
print("Create directory for the experiment (if not already existing)")
if os.path.exists(result_dir):
    print("Dir already existing, Do you want to continue?")
    answer = raw_input()
    if answer == "y" or answer == 'yes':
        shutil.rmtree(result_dir)
        pass
    else:
        sys.exit()


os.mkdir(result_dir)
if fine_tuning == 1:
    print("copy the pretrained model to experiment folder")
    model_path = '../../models/{}_weight.h5'.format(pretrained_model)
    os.system('cp '+ model_path + ' ' + result_dir + '/' + 'pretrain_weight.h5')
print("copy the configuration file in the experiment folder")
os.system('cp ' + config_file + ' ' + result_dir + '/' + name_experiment + '_' + config_file)


# run the experiment
if nohup:
    print("Run the training on GPU with nohup")
    os.system(
        run_GPU + ' nohup python -u ./nuclei_traing.py ' + config_file +' > ' + result_dir + '/' + name_experiment + '_training.nohup &')
else:
    print("Run the training on GPU (no nohup)")
    os.system(run_GPU + ' python -u ./nuclei_traing.py ' + config_file)
