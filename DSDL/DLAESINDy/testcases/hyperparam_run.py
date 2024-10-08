
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from default_params import default_params
import numpy as np
from aesindy.solvers import DatasetConstructor
from aesindy.training import TrainModel
from aesindy.helper_functions import call_polygon
import wandb

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

def wandb_sweep():
    raw_data = call_polygon('SPY','2016-01-01','2024-06-01','minute',15)
    raw_data = raw_data[['c']]
    raw_data = raw_data.rename(columns={'c':'x'})


    # Hyperparameters
    sweep_config = {
        'method': 'bayes', 
        'bayes': {
            'bayes_initial_samples': 75,
            'exploration_factor': 0.2,
        },
        'metric': {
            'name': 'val_rec_loss',
            'goal': 'minimize'
        },
        "parameters": {
            "num_epochs": {'values': [450]},
            "learning_rate": {'values': [0.1,0.01,0.03,0.001]},
            "latent_dim": {'values': [1,2,3,4]},
            "input_dim": {'values': [50,100,200,400]},
            "poly_order": {'values': [1,2,3,4]},
            "include_sine": {'values': [True, False]},
            "loss_weight_layer_l2": {'values': [.0,0.1]},
            "loss_weight_x0": {'values': [0,0.01]},
            "loss_weight_integral": {'values': [0,0.05,0.1]},
            "loss_weight_sindy_regularization": {'values': [1e-5,1e-3]},
            "loss_weight_rec": {'values': [0.3,0.5]},
            "batch_size": {'values': [32,128]},
            "data_length": {'values': [0,.25,.5,.75]},
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="DLAESINDy")

    # Define the objective function for the sweep
    def sweep_train():

        wandb.init(project="DLAESINDy")
        model_runner(wandb.config, raw_data)

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train, count=500) 


if __name__ == '__main__':
    wandb_sweep()