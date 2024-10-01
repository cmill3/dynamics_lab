
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from aesindy.solvers import DatasetConstructor
from aesindy.training import TrainModel
from default_params import params
from aesindy.helper_functions import call_polygon

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


params['model'] = 'spy'
params['case'] = 'test_2nd_dim256_ld2'
params['input_dim'] = 128
params['tend'] = 2
params['poly_order'] = 2
params['include_sine'] = False
params['fix_coefs'] = False
params['svd_dim'] = 2
params['latent_dim'] = 2
params['scale'] = False

params['save_checkpoints'] = True 
params['save_freq'] = 5 

params['print_progress'] = True
params['print_frequency'] = 10

# training time cutoffs
params['max_epochs'] = 300
params['patience'] = 45

# loss function weighting
params['loss_weight_rec'] = 0.3
params['loss_weight_sindy_z'] = 0.001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.1  
params['loss_weight_x0'] = 0.01
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 

raw_data = call_polygon('SPY','2020-01-01','2024-07-01','hour',1)
print(raw_data.head())
print(raw_data.tail())
raw_data = raw_data[['c']]
raw_data = raw_data.rename(columns={'c':'x'})
data_dict = {
    'x':[raw_data['x'].values],
    'dt': 3600
    }

data_builder = DatasetConstructor(input_dim=params['input_dim'],
                interpolate=params['interpolate'],
                interp_dt=params['interp_dt'],
                savgol_interp_coefs=params['interp_coefs'],
                interp_kind=params['interp_kind'])
data_builder.build_solution(data_dict)
train_data = data_builder.get_data()
print(train_data['x'].shape)
print(train_data)
trainer = TrainModel(train_data, params)
trainer.fit() 
