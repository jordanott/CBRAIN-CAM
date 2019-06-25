# tgb - 4/18/2019 - Python script callable from command line
# Follows notebook 010 @ https://github.com/tbeucler/CBRAIN-CAM/blob/master/notebooks/tbeucler_devlog/010_Conserving_Network_Paper_Runs.ipynb
import os
import pprint
gpu = os.environ.get("SHERPA_RESOURCE", '')
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

# set random seeds
import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

import sys
sys.path.append('../')
sys.path.append('../../../')
# import xarray as xr
import numpy as np

# from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.utils import limit_mem
from stored_dictionaries.data_options import data_opts

from model import Network
from cbrain.model_diagnostics import ModelDiagnostics

################## Sherpa trial ##################
import sherpa
client = sherpa.Client()
trial = client.get_trial()  # contains ID and parameters
##################################################

# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(trial.parameters)

if trial.parameters['data'] == 'fluxbypass_aqua':
    PREFIX = '8col009_01_'
    DATADIR = trial.parameters['data_dir'] + trial.parameters['data'] + '/'

    scale_dict = load_pickle(DATADIR + '009_Wm2_scaling.pkl'); in_vars = load_pickle(DATADIR + '009_Wm2_in_vars.pkl')
    out_vars = load_pickle(DATADIR + '009_Wm2_out_vars.pkl'); dP = load_pickle(DATADIR + '009_Wm2_dP.pkl')

    train_gen = DataGenerator(
        data_fn = DATADIR+PREFIX+'train.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = DATADIR+PREFIX+'norm.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict,
        batch_size=trial.parameters['batch_size'],
        shuffle=True
    )

    valid_gen = DataGenerator(
        data_fn = DATADIR+PREFIX+'valid.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = DATADIR+PREFIX+'norm.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict,
        batch_size=trial.parameters['batch_size'],
        shuffle=False
    )

    net = Network(trial.parameters, trial.id,
    	scale_dict=scale_dict,
    	sub=train_gen.input_transform.sub,
    	div=train_gen.input_transform.div
    )
else:
    from data_generator import DataGenerator

    train_gen = DataGenerator(
        data_dir=trial.parameters['data_dir'] + trial.parameters['data'] + '/',
        feature_fn=data_opts[trial.parameters['data']]['train']['feature_fn'],
        target_fn=data_opts[trial.parameters['data']]['train']['target_fn'],
        batch_size=trial.parameters['batch_size'],
        norm_fn=data_opts[trial.parameters['data']]['norm_fn'],
        fsub='feature_means',
        fdiv='feature_stds',
        tmult='target_conv',
        shuffle=True,
    )

    valid_gen = DataGenerator(
        data_dir=trial.parameters['data_dir'] + trial.parameters['data'] + '/',
        feature_fn=data_opts[trial.parameters['data']]['test']['feature_fn'],
        target_fn=data_opts[trial.parameters['data']]['test']['target_fn'],
        batch_size=trial.parameters['batch_size'],
        norm_fn=data_opts[trial.parameters['data']]['norm_fn'],
        fsub='feature_means',
        fdiv='feature_stds',
        tmult='target_conv',
        shuffle=False,
    )

    net = Network(trial.parameters, trial.id)

if trial.parameters['alg'] == 'pbt':
    output_path = 'SherpaResults/{data}_{alg}/{net_type}_{loss_type}/output/'.format(
        data=trial.parameters['data'],
        net_type=trial.parameters['net_type'],
        loss_type=trial.parameters['loss_type'],
        alg=trial.parameters['alg']
    )

    if trial.parameters['load_from'] == '':
        pass
    else:
        net.load(output_path + str(trial.parameters['load_from']).replace('.h5','') + '.h5')
        K.set_value(net.model.optimizer.lr, trial.parameters['lr'])

# save lr model
net.save()

# train linear regression model
net.train(train_gen, valid_gen, trial=trial, client=client)

if trial.parameters['alg'] == 'pbt':
    output_path = 'SherpaResults/{data}_{alg}/{net_type}_{loss_type}/output/'.format(
        data=trial.parameters['data'],
        net_type=trial.parameters['net_type'],
        loss_type=trial.parameters['loss_type'],
        alg=trial.parameters['alg']
    )
    net.save(output_path + str(trial.parameters['save_to']).replace('.h5', ''))

print ('Done with trial ' + str(trial.id) )
