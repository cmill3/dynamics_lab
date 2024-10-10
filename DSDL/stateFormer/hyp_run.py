from layers.patchTST import PatchTST
from dataset_builders import build_dataset
from trainer import Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import wandb

def build_optimizer(model, model_hyperparameters):

    ## check a conditional for optiimizer type between ADAM,rmsprop,sgd
    if model_hyperparameters['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=model_hyperparameters["learning_rate"])
    elif model_hyperparameters['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=model_hyperparameters["learning_rate"])
    elif model_hyperparameters['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=model_hyperparameters["learning_rate"])

    return optimizer

def run_model(model_hyperparameters, train_data, val_data):
    model_hyperparameters['numerical_features'] = feature_params['numerical_features']
    model_hyperparameters['use_wandb'] = True
    model_hyperparameters['kernel_size'] = 25
    model_hyperparameters['decomposition'] = False


    model = PatchTST(model_hyperparameters)
    trainer = Trainer()
    optimizer = build_optimizer(model, model_hyperparameters)
    trainer.setup(model, optimizer)
    trainer.train(train_data, val_data, model_hyperparameters)

feature_params = {
    "numerical_features": ['c','v','range_vol'], # List of numerical features
}

def wandb_sweep():
# Hyperparameters
    sweep_config = {
        'method': 'bayes', 
        'bayes': {
            'bayes_initial_samples': 10,
            'exploration_factor': 0.2,
        },
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        "parameters": {
            "num_epochs": {'values': [200]},
            "batch_size": {'values': [128]},
            "learning_rate": {'values': [0.01, 0.001]},
            "state_space_embedding": {'values': [64,128,256]},
            "attention_heads": {'values': [8,16,32]},
            "hidden_state_dim": {'values': [64,128,256,512]},
            "feedforward_dim": {'values': [64,128,256,512]},
            "dropout": {'values': [.05,0.1]},
            "transformer_layers": {'values': [4,16,32,64]},
            "patch_len": {'values': [8, 16, 32, 48]},
            "stride": {'values': [.25, .5]},
            "batches_per_epoch": {'values': [100]},
            "prediction_horizon": {'values': [4]},
            "optimizer": {'values': ['adam']},
            "patience": {'values': [20]},
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="patchTST_state")

    # Define the objective function for the sweep
    def sweep_train():
        wandb.init(project="patchTST_trend")
        feature_data, target_data = build_dataset(wandb.config, local_dataset="/Users/charlesmiller/Documents/Code/dynamics_lab/DSDL/stateFormer/scaled_data_SPY.csv")
        run_model(wandb.config, feature_data, target_data)

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train, count=45)  # Run 10 trials

if __name__ == "__main__":
    wandb_sweep()
