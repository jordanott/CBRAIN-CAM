import os
import netCDF4
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from utils import build_directory
from data_generator import DataGenerator
from stored_dictionaries.data_options import data_opts

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['fluxbypass_aqua', 'land_data', '8col', '32col'])
parser.add_argument('--hp_pred_path', type=str)
parser.add_argument('--baseline_pred_path', type=str)
parser.add_argument('--out_path', type=str)
parser.add_argument('--rescale', action='store_true')
args = parser.parse_args()

args.baseline_pred_path = '/baldig/chemistry/earth_system_science/{data}/'.format(data=args.data) + args.baseline_pred_path
args.hp_pred_path = '/baldig/chemistry/earth_system_science/{data}/'.format(data=args.data) + args.hp_pred_path

args.out_path += 'Plots/'
build_directory(args.out_path)

# create mesh grid
mesh_info = netCDF4.Dataset('/baldig/chemistry/earth_system_science/grid_for_griffin.nc')
lats = np.array(mesh_info.variables['lat']); levs = np.array(mesh_info.variables['lev'])
Xvar, Yvar = np.meshgrid(levs, lats)

def mesh_plot(R2, start_idx=0, end_idx=30, file_name=''):
    if start_idx == 0: title = 'Heating Rate SPCAM3 $R^2$ Error Map'
    else: title = 'Moistening Rate SPCAM3 $R^2$ Error Map'

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.flip(R2[:,start_idx:end_idx].T), cmap = 'RdYlBu', vmin=-1, vmax=1,
        interpolation='bilinear',
        extent=[np.min(lats), np.max(lats), np.min(levs), np.max(levs)])
    fig.colorbar(im, label='R^2'); plt.title(title, fontsize=13)
    plt.xlabel('Latitude',fontsize=20); plt.ylabel('Pressure', fontsize=20)
    plt.gca().set_yticks([200,400,600,800]); plt.gca().set_xticks([-80,-60,-40,-20, 0, 20, 40, 60, 80])
    plt.gca().invert_yaxis(); ax.set_aspect('auto')
    plt.savefig(args.out_path+file_name)

def r2_vs_pressure(old, hp, levs, title, file_name):
    fig, ax = plt.subplots()
    plt.plot(old, levs, label='Model by Rasp et al.')
    plt.plot(hp, levs, '--', label='SHERPA')

    plt.xlabel('$R^2$',fontsize=20); plt.ylabel('Pressure', fontsize=20)
    plt.title(title+' $R^2$ vs Pressure', fontsize=13); plt.gca().invert_yaxis()
    plt.legend(); plt.xlim(0, 1); plt.tight_layout()
    plt.savefig(args.out_path+file_name)

def r2_vs_lat(old, hp, lats, title, file_name):
    fig, ax = plt.subplots()
    plt.plot(lats, old, label='Model by Rasp et al.')
    plt.plot(lats, hp, '--', label='SHERPA')

    plt.ylabel('$R^2$',fontsize=20); plt.xlabel('Latitude', fontsize=20)
    plt.title(title+' Latitude vs $R^2$', fontsize=13)
    plt.legend(); plt.tight_layout()
    plt.savefig(args.out_path+file_name)

def get_R2(predictions_file, hp=False):
    targets_dataset = netCDF4.Dataset('/baldig/chemistry/earth_system_science/{data}/{targets}'.format(
        data=args.data,
        targets=data_opts[args.data]['test']['target_fn']
    ))

    targets = np.array(targets_dataset['targets'])

    predictions = np.load(predictions_file)['predictions']
    if args.rescale:
        valid_gen = DataGenerator(
            data_dir='/baldig/chemistry/earth_system_science/8col/',
            feature_fn='full_physics_essentials_valid_month02_features.nc',
            target_fn='full_physics_essentials_valid_month02_targets.nc',
            batch_size=512,
            norm_fn='full_physics_essentials_train_month01_norm.nc',  # SAME NORMALIZATION FILE!
            fsub='feature_means',
            fdiv='feature_stds',
            tmult='target_conv',
            shuffle=False,
        )

        predictions = (predictions/valid_gen.target_norms[1]) + valid_gen.target_norms[0]
    # flatten arrays
    predictions = predictions.reshape(-1,65); targets = targets.reshape(-1, 65)

    lon_dim = 128 # x
    lat_dim = 64 # y
    sample_dim = predictions.reshape(128,64,-1,65).shape[2]
    heat_dim = 65 # z

    reconstructed_targets = np.zeros(shape=(lon_dim, lat_dim, sample_dim, heat_dim))
    reconstructed_predictions = np.zeros(shape=(lon_dim, lat_dim, sample_dim, heat_dim))

    count = 0
    for sample in range(sample_dim):
        for lat in range(lat_dim):
            for lon in range(lon_dim):
                reconstructed_targets[lon,lat, sample, :] = targets[count]
                reconstructed_predictions[lon,lat, sample, :] = predictions[count]
                count += 1

    sse = np.sum((reconstructed_targets - reconstructed_predictions)**2, axis=2)
    mean_targets = np.mean(reconstructed_targets, axis=2)[:,:,np.newaxis]

    svar = np.sum((reconstructed_targets - mean_targets)**2, axis=2)
    R_2 = 1 - (sse/svar)

    R_2 = np.nanmean(R_2, axis = 0)
    # clamp values
    R_2 = np.clip(R_2, -1, 1)
    return R_2
# '/baldig/chemistry/earth_system_science/8col/normal_mse/predictions_6.npz'
hp_model_R2 = get_R2(args.hp_pred_path, hp=True)
mesh_plot(hp_model_R2, start_idx=0, end_idx=30, file_name='hp_heating_rate.png')
mesh_plot(hp_model_R2, start_idx=30, end_idx=60, file_name='hp_moistening_rate.png')

print('Done plotting hp')
old_model_R2 = get_R2(args.baseline_pred_path)

mesh_plot(old_model_R2, start_idx=0, end_idx=30, file_name='old_model_heating_rate.png')
mesh_plot(old_model_R2, start_idx=30, end_idx=60, file_name='old_moistening_rate.png')

import seaborn as sns; sns.set()

old = np.nanmean(old_model_R2[:,:30], axis=0); hp = np.nanmean(hp_model_R2[:,:30], axis=0)
r2_vs_pressure(old, hp, levs, 'Convective Heating Rate', 'heating_rate.png')

old = np.nanmean(old_model_R2[:,30:60], axis=0); hp = np.nanmean(hp_model_R2[:,30:60], axis=0)
r2_vs_pressure(old, hp, levs, 'Convective Moistening Rate', 'moistening_rate.png')

old = old_model_R2[:,60]; hp = hp_model_R2[:,60]
r2_vs_lat(old, hp, lats, 'Shortwave Flux at TOA', 'shortwave_flux_at_toa.png')

old = old_model_R2[:,61]; hp = hp_model_R2[:,61]
r2_vs_lat(old, hp, lats, 'Shortwave Flux at Surface', 'shortwave_flux_at_surface.png')

old = old_model_R2[:,62]; hp = hp_model_R2[:,62]
r2_vs_lat(old, hp, lats, 'Longwave Flux at TOA', 'longwave_flux_at_toa.png')

old = old_model_R2[:,63]; hp = hp_model_R2[:,63]
r2_vs_lat(old, hp, lats, 'Longwave Flux at Surface', 'longwave_flux_at_surface.png')

old = old_model_R2[:,64]; hp = hp_model_R2[:,64]
r2_vs_lat(old, hp, lats, 'Precipitation', 'precipitation.png')
