import os
import pprint
import argparse
import itertools
from utils import build_directory
from stored_dictionaries.default import default_params

# set random seeds
import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

import sys
sys.path.append('../')
sys.path.append('../../../')
import numpy as np

# from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.utils import limit_mem
from stored_dictionaries.data_options import data_opts

from model import Network
from cbrain.model_diagnostics import ModelDiagnostics

parser = argparse.ArgumentParser()
# ---------------- Important parameters -------------------------
parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'weak_loss'], help='What to run?')
parser.add_argument('--net_type', type=str, default='normal', choices=['normal', 'conservation'], help='What to run?')
parser.add_argument('--data', type=str, choices=['fluxbypass_aqua', 'land_data', '8col', '32col'])
# params okay left as defaults
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--data_dir', type=str, default='/baldig/chemistry/earth_system_science/')
parser.add_argument('--epochs', type=int, default=18, help='Number of epochs used for training')
parser.add_argument('--patience', type=int, default=10, help='How long to wait for an improvement')
parser.add_argument('--alg', default='baseline')
FLAGS = parser.parse_args()

output_path = 'SherpaResults/baselines/{data}/{net_type}_{loss_type}/'.format(
    data=FLAGS.data,
    net_type=FLAGS.net_type,
    loss_type=FLAGS.loss_type,
)
models_path = output_path + 'Models/'

build_directory(models_path)

# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()

args = vars(FLAGS)
args.update(default_params[FLAGS.data])

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(args)

if args['data'] == 'fluxbypass_aqua':
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

    net = Network(args, 1,
    	scale_dict=scale_dict,
    	sub=train_gen.input_transform.sub,
    	div=train_gen.input_transform.div
    )
else:
    from data_generator import DataGenerator

    train_gen = DataGenerator(
        data_dir=args['data_dir'] + args['data'] + '/',
        feature_fn=data_opts[args['data']]['train']['feature_fn'],
        target_fn=data_opts[args['data']]['train']['target_fn'],
        batch_size=args['batch_size'],
        norm_fn=data_opts[args['data']]['norm_fn'],
        fsub='feature_means',
        fdiv='feature_stds',
        tmult='target_conv',
        shuffle=True,
    )

    valid_gen = DataGenerator(
        data_dir=args['data_dir'] + args['data'] + '/',
        feature_fn=data_opts[args['data']]['test']['feature_fn'],
        target_fn=data_opts[args['data']]['test']['target_fn'],
        batch_size=args['batch_size'],
        norm_fn=data_opts[args['data']]['norm_fn'],
        fsub='feature_means',
        fdiv='feature_stds',
        tmult='target_conv',
        shuffle=False,
    )

    net = Network(args, 1)

# save lr model
net.save()
net.train(train_gen, valid_gen)
