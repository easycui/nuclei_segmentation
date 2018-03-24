###################################################
#
#   Script to execute the prediction
#
##################################################
from __future__ import print_function
import os, sys

from utils.help_functions import parse_config


#config file to read from
if len(sys.argv)>1:
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print('The config file does not exist')
        sys.exit()
else:
    config_file = './configuration.txt'
config = parse_config(config_file)
#===========================================
#name of the experiment!!
name_experiment = config['name']
nohup = config['test_nohup']   #std output on log file?

run_GPU = ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results if not existing already
result_dir = './experiment/' + name_experiment
print("experiment:", result_dir)
print("Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    pass
else:
    print('this experiment does not exist')
    sys.exit()

# finally run the prediction
if nohup:
    print("Run the prediction on GPU  with nohup")
    os.system(run_GPU +' nohup python -u ./nuclei_test.py ' + config_file +' > ' + result_dir + '/' + name_experiment+'_prediction.nohup')
else:
    print("Run the prediction on GPU (no nohup)")
    os.system(run_GPU +' python -u ./nuclei_test.py ' + config_file)
