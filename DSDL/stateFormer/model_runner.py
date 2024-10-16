from layers.patchTST import PatchTST
from dataset_builders import build_dataset
from trainer import Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

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
    model = PatchTST(model_hyperparameters)
    trainer = Trainer()
    optimizer = build_optimizer(model, model_hyperparameters)
    trainer.setup(model, optimizer)
    trainer.train(train_data, val_data, model_hyperparameters)

model_hyperparameters = {
    "state_space_embedding": 256 , # Dimension of your input features
    # "embedding_dim": 512,   # Dimension of the model
    "prediction_horizon": 4, # How many steps ahead you want to predict
    "numerical_features": ['c','v','range_vol'], # List of numerical features
    "attention_heads": 16,
    "hidden_state_dim": 256,
    "feedforward_dim": 256,
    "dropout": 0.05,
    "transformer_layers": 16,
    "individual": True,
    "patch_len": 32,
    "stride": 16,
    "padding_patch": 'end',
    "revin": False,
    "affine": False,
    "patience": 20,
    "kernel_size": 25,
    "subtract_last": 0,
    "optimizer": "adam",
    "categorical": True,
    "decomposition": False,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "batches_per_epoch": 300,
    "use_wandb": False,
}

if __name__ == '__main__':
    feature_data, target_data = build_dataset(model_hyperparameters, local_dataset=None)
    run_model(model_hyperparameters, feature_data, target_data)
