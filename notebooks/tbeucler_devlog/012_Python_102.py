# tgb - 5/1/2019 - Calculates statistics on NNLA for all datasets
# tgb - 4/27/2019 - Calculates precipitation PDF for each network on +0K and +4K
# tgb - 4/24/2019 - Data-scarce using just 1-file --> calculate statistics on +0,1,2,3,4K; NNA version
# tgb - 4/24/2019 - Validate the unconstrained multiple linear regression model on +1,2,3,4K
# tgb - 4/22/2019 - Use +1K as validation dataset
# tgb - 4/19/2019 - The goal is to make a slurm-callable script to calculate the statistics and residuals of all the paper neural networks over the validation dataset. This script is specialized to the +0K experiment.

from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()
TRAINDIR = '/local/Tom.Beucler/SPCAM_PHYS/'
DATADIR = '/project/meteo/w2w/A6/S.Rasp/SP-CAM/fluxbypass_aqua/'

import os
os.chdir('/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM')

config_fn = '/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM/pp_config/8col_rad_tbeucler_local_PostProc.yml'
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer}
data_fn_array = ['/local/Tom.Beucler/SPCAM_PHYS/8col009_01_valid.nc',
                 '/local/Tom.Beucler/SPCAM_PHYS/8col009_11_valid.nc',
                 '/local/Tom.Beucler/SPCAM_PHYS/8col009_12_valid.nc',
                 '/local/Tom.Beucler/SPCAM_PHYS/8col009_13_valid.nc',
                 '/local/Tom.Beucler/SPCAM_PHYS/8col009_14_valid.nc']
dataref = ['','1K','2K','3K','4K']

NN = {}
print('Loading model') # 1) Load model
NN = load_model(TRAINDIR+'HDF5_DATA/'+'NNA0.01'+'.h5',custom_objects=dict_lay)    
for j in range(len(data_fn_array)):
    md = {}
    print('j=',j)
    print('Loading statistics') # 2) Define model diagnostics object
    md = ModelDiagnostics(NN,config_fn,data_fn_array[j])
    # 3) Calculate statistics and save in pickle file
    md.compute_stats()
    pickle.dump(md.stats,open(TRAINDIR+'HDF5_DATA/'+'NNA0.01'+
                               'md'+dataref[j]+'.pkl','wb'))
