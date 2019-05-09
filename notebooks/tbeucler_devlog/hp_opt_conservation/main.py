# tgb - 4/18/2019 - Python script callable from command line
# Follows notebook 010 @ https://github.com/tbeucler/CBRAIN-CAM/blob/master/notebooks/tbeucler_devlog/010_Conserving_Network_Paper_Runs.ipynb

# set random seeds
import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

import sys
sys.path.append('../')
sys.path.append('../../../')

import os
import argparse
import xarray as xr
import numpy as np

from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.utils import limit_mem

from model import Network
from monitor import MetricMonitor
from cbrain.model_diagnostics import ModelDiagnostics

# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()

parser = argparse.ArgumentParser()
# important params
parser.add_argument('--data', type=str, default='fluxbypass_aqua', choices=['fluxbypass_aqua'])
parser.add_argument('--continue', default=False, action='store_true', help='Continue from saved')
parser.add_argument('--get_pred', default=False, action='store_true', help='Run predictions for specified model')
parser.add_argument('--run_type', type=str, default='hyper_param_opt', choices=['hyper_param_opt', 'baseline', 'hyper_param_opt_conservation'], help='What to run?')
# params okay left as defaults
parser.add_argument('--batch_size', type=int, default=8192, help='Batch size')
parser.add_argument('--data_dir', type=str, default='/baldig/chemistry/earth_system_science/')
parser.add_argument('--max_dense_layers', type=int, default=5, help='Max dense layers allowed')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs used for training')
parser.add_argument('--patience', type=int, default=8, help='How long to wait for an improvement')

args = vars(parser.parse_args())

metric_monitor = MetricMonitor(args)

PREFIX = '8col009_01_'
DATADIR = args['data_dir'] + args['data'] + '/'

scale_dict = load_pickle(DATADIR + '009_Wm2_scaling.pkl'); in_vars = load_pickle(DATADIR + '009_Wm2_in_vars.pkl')
out_vars = load_pickle(DATADIR + '009_Wm2_out_vars.pkl'); dP = load_pickle(DATADIR + '009_Wm2_dP.pkl')


train_gen = DataGenerator(
    data_fn = DATADIR+PREFIX+'train.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = DATADIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=args['batch_size'],
    shuffle=True
)

args['div'] = train_gen.input_transform.div; args['sub'] = train_gen.input_transform.sub
args['scale_dict'] = scale_dict

valid_gen = DataGenerator(
    data_fn = DATADIR+PREFIX+'valid.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = DATADIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=args['batch_size'],
    shuffle=False
)

if args['run_type'] == 'baseline':
    net = Network(args)
    # train linear regression model
    history = net.train(train_gen, valid_gen)
    # save training results
    metric_monitor.update_trial_storage(history)
    # save lr model
    net.save(weights=True)
    # store predictions from lr model
    net.predict(valid_gen, file_name=metric_monitor.get_pred_loc())

elif 'hyper_param_opt' in args['run_type']:
    # iterate through sherpa trials
    for params, (trial, study) in metric_monitor.run_hyper_param_opt():
        try:
            # create new network with hyper params each trial
            net = Network(params)
            # fit the data
            history = net.train(train_gen, valid_gen, trial=trial, study=study)

            # save current model configuration
            net.save()

            # record results and param settings
            best_loss = metric_monitor.update_trial_storage(history)

            if best_loss:
                # store predictions made by current model
                net.predict(valid_gen, file_name=metric_monitor.get_pred_loc())
            else:
                # remove weights file if the loss hasn't improved
                os.remove(net.get_model_path()+'.h5')

            metric_monitor.end_trial()

        except Exception as e:
            # clear memory from keras
            K.clear_session()
