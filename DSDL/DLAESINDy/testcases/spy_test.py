
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


params['model'] = 'custom'
params['case'] = 'custom'
params['system_coefficients'] = [9.8, 10]
params['noise'] = 0.0
params['input_dim'] = 80
params['dt'] = np.sqrt(params['system_coefficients'][0]/params['system_coefficients'][1])/params['input_dim']/5
params['tend'] = 2
params['n_ics'] = 30
params['poly_order'] = 1
params['include_sine'] = True
params['fix_coefs'] = False

params['save_checkpoints'] = True 
params['save_freq'] = 5 

params['print_progress'] = True
params['print_frequency'] = 5 

# training time cutoffs
params['max_epochs'] = 300
params['patience'] = 25

# loss function weighting
params['loss_weight_rec'] = 0.3
params['loss_weight_sindy_z'] = 0.001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.1  
params['loss_weight_x0'] = 0.01 
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 

raw_data = call_polygon('SPY','2024-01-01','2024-07-01','minute',15)
raw_data = raw_data[['c']]
raw_data = raw_data.rename(columns={'c':'x'})
data_dict = {
    'x':[raw_data['x'].values],
    'dt': 900
    }

data_builder = DatasetConstructor(input_dim=params['input_dim'],
                interpolate=params['interpolate'],
                interp_dt=params['interp_dt'],
                savgol_interp_coefs=params['interp_coefs'],
                interp_kind=params['interp_kind'])
train_data = data_builder.build_solution(data_dict)
trainer = TrainModel(train_data, params)
trainer.fit() 
