# tgb - 4/23/2019 - Only loops over the last values of alpha and adding the architecture-constrained network
# tgb - 4/22/2019 - Use +1K as validation dataset
# tgb - 4/19/2019 - The goal is to make a slurm-callable script to calculate the statistics and residuals of all the paper neural networks over the validation dataset. This script is specialized to the +0K experiment.

import sys
sys.path.append('../')
sys.path.append('../../../')

import json
import argparse
from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
from tensorflow.keras.models import load_model
from utils import build_directory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics

# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/baldig/chemistry/earth_system_science/')
parser.add_argument('--data', type=str, default='fluxbypass_aqua', choices=['fluxbypass_aqua'])
parser.add_argument('--net_type', type=str, default='normal', choices=['normal', 'conservation'], help='What to run?')
parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'weak_loss'], help='What to run?')
parser.add_argument('--diagnostics', default=False, action='store_true', help='Want to run model diagnostics?')
FLAGS = parser.parse_args()

model_path = 'SherpaResults/{data}/{net_type}_{loss_type}/Models/'.format(
    data=FLAGS.data,
    net_type=FLAGS.net_type,
    loss_type=FLAGS.loss_type
)
output_path = 'SherpaResults/{data}/{net_type}_{loss_type}/Diagnostics/'.format(
    data=FLAGS.data,
    net_type=FLAGS.net_type,
    loss_type=FLAGS.loss_type
)
plots_path = output_path + 'Plots/'

build_directory(output_path); build_directory(plots_path)

DATADIR = FLAGS.data_dir + FLAGS.data + '/'
data_fn = DATADIR + '8col009_01_valid.nc'

config_fn = '../../../pp_config/8col_rad_tbeucler_local_PostProc.yml'
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer}

for model_file in os.listdir(model_path):
    if model_file.endswith('h5') and FLAGS.diagnostics:
        model_number = model_file.replace('.h5', '')

        # 1) Load model
        NN = load_model(model_path + model_file, custom_objects=dict_lay)

        # 2) Define model diagnostics object
        md = ModelDiagnostics(NN,config_fn,data_fn)

        # 3) Calculate statistics and save in pickle file
        md.compute_stats()
        pickle.dump(md.stats,open(output_path + model_file + '_md1K.pkl','wb'))

        # 4) Calculate budget residuals and save in pickle file
        md.compute_res()
        pickle.dump(md.res,open(output_path + model_file + '_res1K.pkl','wb'))
    if model_file.endswith('json'):
        model_number = model_file.replace('json', '')

        with open(model_path + model_file, 'r') as f:
            loss_info = eval(f.read())

        plt.clf()
        plt.plot(loss_info['loss'], label='Loss')
        plt.plot(loss_info['val_loss'], label='Val Loss')
        if FLAGS.loss_type == 'weak_loss':
            plt.plot(loss_info['mean_squared_error'], label='MSE')
            plt.plot(loss_info['val_mean_squared_error'], label='Val MSE')

        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.title('Net type: {net_type} Loss type: {loss_type}'.format(
            net_type=FLAGS.net_type,
            loss_type=FLAGS.loss_type
        ))
        plt.legend()
        plt.savefig(plots_path + model_number + 'png')
