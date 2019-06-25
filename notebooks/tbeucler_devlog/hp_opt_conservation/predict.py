# set random seeds
import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

import os
import sys
import argparse
sys.path.append('../')
sys.path.append('../../../')

from cbrain.data_generator import *
from cbrain.losses import *
from cbrain.layers import *
from cbrain.cam_constants import *

from utils import build_directory
from cbrain.utils import limit_mem
from stored_dictionaries.data_options import data_opts
from tensorflow.keras.models import *

parser = argparse.ArgumentParser()
# python3.6 predict.py --loss_type mse --net_type normal --data 8col --model_path
# ---------------- Important parameters -------------------------
parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'weak_loss'], help='What to run?')
parser.add_argument('--net_type', type=str, default='normal', choices=['normal', 'conservation'], help='What to run?')
parser.add_argument('--data', type=str, choices=['fluxbypass_aqua', 'land_data', '8col', '32col'])
parser.add_argument('--model_path', type=str)
parser.add_argument('--alg', type=str, default='')

# params okay left as defaults
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--data_dir', type=str, default='/baldig/chemistry/earth_system_science/')
parser.add_argument('--patience', type=int, default=10, help='How long to wait for an improvement')

FLAGS = parser.parse_args()

# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()
output_path = '{data_dir}{data}/{net_type}_{loss_type}/'.format(
    data_dir=FLAGS.data_dir,
    data=FLAGS.data,
    net_type=FLAGS.net_type,
    loss_type=FLAGS.loss_type,
    alg=FLAGS.alg
); build_directory(output_path)

if FLAGS.data == 'fluxbypass_aqua':
    PREFIX = '8col009_01_'
    DATADIR = FLAGS.data_dir + FLAGS.data + '/'

    scale_dict = load_pickle(DATADIR + '009_Wm2_scaling.pkl'); in_vars = load_pickle(DATADIR + '009_Wm2_in_vars.pkl')
    out_vars = load_pickle(DATADIR + '009_Wm2_out_vars.pkl'); dP = load_pickle(DATADIR + '009_Wm2_dP.pkl')

    valid_gen = DataGenerator(
        data_fn = DATADIR+PREFIX+'valid.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = DATADIR+PREFIX+'norm.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict,
        batch_size=FLAGS.batch_size,
        shuffle=False
    )
else:
    from data_generator import DataGenerator

    valid_gen = DataGenerator(
        data_dir=FLAGS.data_dir + FLAGS.data + '/',
        feature_fn=data_opts[FLAGS.data]['test']['feature_fn'],
        target_fn=data_opts[FLAGS.data]['test']['target_fn'],
        batch_size=FLAGS.batch_size,
        norm_fn=data_opts[FLAGS.data]['norm_fn'],
        fsub='feature_means',
        fdiv='feature_stds',
        tmult='target_conv',
        shuffle=False,
    )

    # load weights from h5 file
    model = load_model(FLAGS.model_path)

    trial_num = FLAGS.model_path.split('/')[-1].replace('.h5','')

    print('Trial Num:', trial_num)

    predictions = model.predict_generator(
        valid_gen.return_generator(),
        steps=valid_gen.n_batches,
        workers=16,
        max_queue_size=50,
    )

    # rescale predictions
    predictions = (predictions/valid_gen.target_norms[1]) + valid_gen.target_norms[0]

    # save predictions to path
    np.savez(output_path + 'predictions_{}.npz'.format(trial_num), predictions=predictions)
