
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from default_params import default_params
import numpy as np
from aesindy.solvers import DatasetConstructor
from aesindy.training import TrainModel
from aesindy.helper_functions import call_polygon
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

def update_params_from_wandb(params, wandb_config):
    """
    Update the default parameters with values from the wandb config.
    
    Args:
    default_params (dict): The default parameter dictionary
    wandb_config (wandb.config): The wandb config object
    
    Returns:
    dict: Updated parameter dictionary
    """
    updated_params = params.copy()
    
    for key, value in wandb_config.items():
        if key in updated_params:
            updated_params[key] = value
        else:
            print(f"Warning: '{key}' found in wandb config but not in default params. Ignoring.")
    
    return updated_params



def model_runner(wandb_params, raw_data):
    params = update_params_from_wandb(default_params, wandb_params)
    params['model'] = 'spy'
    params['case'] = '1hr_3rd_dim64_ld3_sine_x001'
    params['use_wandb'] = True
    print(params)
    ## slice the data based on a fractional proportion, must remain in sequential order
    raw_data = raw_data[int(params['data_length']*len(raw_data)):]
    data_dict = {
        'x':[raw_data['x'].values],
        'dt': 900
        }
    data_builder = DatasetConstructor(input_dim=params['input_dim'],
                    interpolate=params['interpolate'],
                    interp_dt=params['interp_dt'],
                    savgol_interp_coefs=params['interp_coefs'],
                    interp_kind=params['interp_kind'])
    data_builder.build_solution(data_dict)
    train_data = data_builder.get_data()
    trainer = TrainModel(train_data, params)
    trainer.fit() 

def wandb_sweep(data):


    # Hyperparameters
    sweep_config = {
        'method': 'bayes', 
        'bayes': {
            'bayes_initial_samples': 75,
            'exploration_factor': 0.2,
        },
        'metric': {
            'name': 'current_best_val_rec_loss',
            'goal': 'minimize'
        },
        "parameters": {
            "learning_rate": {'values': [0.003,0.001,.0001]},
            "latent_dim": {'values': [2,3,4,5,8]},
            "input_dim": {'values': [64,128,256]},
            "poly_order": {'values': [2,3]},
            "include_fourier": {'values': [True, False]},
            "n_frequencies": {'values': [2,3,4]},
            "loss_weight_layer_l2": {'values': [.0,0.05]},
            "loss_weight_x0": {'values': [0.01,0.05]},
            "loss_weight_integral": {'values': [0.01,0.05,0.1]},
            "loss_weight_sindy_regularization": {'values': [1e-5,1e-3,1e-1]},
            "loss_weight_rec": {'values': [0.3,0.6,0.9]},
            "loss_weight_sindy_z": {'values': [0.001,0.0001]},
            "loss_weight_sindy_x": {'values': [0.001,0.0001]},
            "batch_size": {'values': [32,128]},
            "data_length": {'values': [0,.25,.5,.75]},
            "widths_ratios": {'values': [[0.5,0.25],[0.75,0.5,0.25],[0.8,0.6,0.4,0.2]]},
            "activation": {'values': ['elu','relu']},
            "use_bias": {'values': [True, False]},
            "sindy_threshold": {'values': [0.01,0.1,0.2,0.3]},
            "sindy_init_scale": {'values': [3.0,5.0,7.0,10.0]},
            "threshold_frequency": {'values': [10,20,40]},
            "coefficient_threshold": {'values': [0.5,1,2,3,4]},
            "sindycall_freq": {'values': [20,50,75]},
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="DLAESINDy")

    # Define the objective function for the sweep
    def sweep_train():

        wandb.init(project="DLAESINDy", config=wandb.config)
        model_runner(wandb.config, data)

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train, count=1000) 


if __name__ == '__main__':
    raw_data = call_polygon('SPY','2012-01-01','2024-06-01','minute',15)
    raw_data = raw_data[['c']]
    raw_data = raw_data.rename(columns={'c':'x'})
    wandb_sweep(raw_data)