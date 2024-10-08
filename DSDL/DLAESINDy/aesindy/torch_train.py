import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.autograd as autograd
import torch.optim as optim
import wandb
from .sindy_utils import library_size, sindy_library
from .torch_network import SindyAutoencoder
import math



class Trainer:
    def __init__(self, data, model_hyperparameters):
        self.data = data
        self.model_hyperparameters = self.fix_params(model_hyperparameters)
        self.device = self.set_device()
        self.model = SindyAutoencoder(self.model_hyperparameters, self.device)
        self.optimizer = self.build_optimizer()


    def get_loss_function(self):
        if self.model_hyperparameters["loss_function"] == "Huber":
            return nn.HuberLoss()
        elif self.model_hyperparameters["loss_function"] == "MSE":
            return nn.MSELoss()
        
    def set_device(self):
        if self.model_hyperparameters["use_metal"]:
            return torch.device("mps")
        else:
            return torch.device("cpu")

        
    def build_optimizer(self):
        ## check a conditional for optiimizer type between ADAM,rmsprop,sgd
        if self.model_hyperparameters['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.model_hyperparameters["learning_rate"])
        elif self.model_hyperparameters['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.model_hyperparameters["learning_rate"])
        elif self.model_hyperparameters['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.model_hyperparameters["learning_rate"])

        return optimizer

        
    def fix_params(self, model_hyperparameters):
        input_dim = model_hyperparameters['input_dim']
        model_hyperparameters['widths'] = [int(i*input_dim) for i in model_hyperparameters['widths_ratios']]
        model_hyperparameters['library_dim'] = library_size(model_hyperparameters['latent_dim'], model_hyperparameters['poly_order'], model_hyperparameters['include_sine'], True)

        # if 'sparse_weighting' in model_hyperparameters:
        #     if model_hyperparameters['sparse_weighting'] is not None:
        #         a, sparse_weights = sindy_library(self.data.z[:100, :], model_hyperparameters['poly_order'], include_sparse_weighting=True)
        #         model_hyperparameters['sparse_weighting'] = sparse_weights

        return model_hyperparameters
    
    def build_dataloaders(self, data_dict, train_size, val_size, test_size, batch_size=32, shuffle=True):
        """
        Build PyTorch DataLoaders from a dictionary of data.

        Args:
        data_dict (dict): A dictionary containing 't', 'x', 'dx', 'z', 'dz', and 'sindy_coefficients'.
        train_test_split (float): Ratio of training data to total data (default: 0.8).
        batch_size (int): Batch size for DataLoaders (default: 32).
        shuffle (bool): Whether to shuffle the data (default: True).

        Returns:
        tuple: (train_loader, test_loader)
        """
        # Convert numpy arrays to PyTorch tensors
        # t = torch.tensor(data_dict['t'], dtype=torch.float32)
        x = torch.tensor(data_dict['x'].T, dtype=torch.float32)
        dx = torch.tensor(data_dict['dx'].T, dtype=torch.float32)
        # z = torch.tensor(data_dict['z'], dtype=torch.float32)
        # dz = torch.tensor(data_dict['dz'], dtype=torch.float32)
        # sindy_coefficients = torch.tensor(data_dict['sindy_coefficients'], dtype=torch.float32)

        # print(f"t shape: {t.shape}")
        # print(f"x shape: {x.shape}")
        # print(f"dx shape: {dx.shape}")
        # print(f"z shape: {z.shape}")
        # print(f"dz shape: {dz.shape}")

        # Create a TensorDataset

        # print(f"Batch size: {batch_size}")
        # print(f"{dataset[0].shape}") 

        # Calculate the split
        train_len = math.floor(train_size * len(x))
        val_len = math.floor(val_size * len(x))

        # Split the dataset
        train_dataset = TensorDataset(x[:train_len], dx[:train_len])
        val_dataset = TensorDataset(x[train_len:train_len+val_len], dx[train_len:train_len+val_len])

        # print(f"train_len: {train_len}, val_len: {val_len}")
        # print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}")
        # print(train_dataset.shape)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if train_size + val_size < 1:
            test_dataset = TensorDataset(x[train_len+val_len:], dx[train_len+val_len:])
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None

        # print(f" train_loader: {len(train_loader)}, val_loader: {len(val_loader)}, test_loader: {len(test_loader)}")
        return train_loader, val_loader, test_loader
    
    def sequential_threshold(self,t):
        if (t % self.model_hyperparameters['seq_thres'] == 0 and t>1):
            self.model.XI_coefficient_mask = torch.abs(self.model.XI) > 0.1
    

    # def get_data(self):
    #     # Split into train and test sets
    #     train_x, test_x = train_test_split(self.data['x'].T, train_size=self.model_hyperparameters['train_ratio'], shuffle=False)
    #     # val_x, test_x = train_test_split(val_x, train_size=self.params['test_ratio'], shuffle=False)
    #     train_dx, test_dx = train_test_split(self.data['dx'].T, train_size=self.model_hyperparameters['train_ratio'], shuffle=False)
    #     # val_dx, test_dx = train_test_split(val_dx, train_size=self.params['test_ratio'], shuffle=False)
    #     train_data = [train_x, train_dx]  
    #     test_data = [test_x, test_dx]  
    #     # val_data = [val_x, val_dx]
    #     if self.model_hyperparameters['svd_dim'] is not None:
    #         train_xorig, test_xorig = train_test_split(self.data['xorig'].T, train_size=self.model_hyperparameters['train_ratio'], shuffle=False)
    #         # val_xorig, test_xorig = train_test_split(val_xorig, train_size=self.params['test_ratio'], shuffle=False)
    #         train_data = [train_xorig] + train_data
    #         test_data = [test_xorig] + test_data 
    #         # val_data = [val_xorig] + val_data
            
    #     return train_data, test_data

    def train(self):
        ## Enable anomaly detection
        autograd.set_detect_anomaly(True)
        self.model.to(self.device)

        loss_list = {}
        loss_list['recon_loss'] = []
        loss_list['sindy_loss_x'] = []
        loss_list['sindy_loss_z'] = []
        loss_list['sindy_regular_loss'] = []
        loss_list['total_loss'] = []
        loss_val_list = []

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        train_loader, val_loader, _ = self.build_dataloaders(self.data, train_size=0.8, val_size=0.15, test_size=0.05,batch_size=self.model_hyperparameters['batch_size'], shuffle=True)

        for epoch in range(self.model_hyperparameters["num_epochs"]):
            epoch_start = time.time()
            epoch_timeout = 1800
            self.model.train()
            loss_epoch = {}
            loss_epoch['recon_loss'] = []
            loss_epoch['sindy_loss_x'] = []
            loss_epoch['sindy_loss_z'] = []
            loss_epoch['sindy_regular_loss'] = []
            loss_epoch['total_loss'] = []
                    

            for batch_idx, (batch_x,batch_x_dt) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.model_hyperparameters['num_epochs']}")):
                if batch_idx >= self.model_hyperparameters["batches_per_epoch"]:
                    break
                elif time.time() - epoch_start > epoch_timeout:
                    print(f"Epoch {epoch+1} timed out")
                    # wandb.log({
                    #     "epoch_status": "timeout", 
                    #     "last_completed_epoch": epoch,
                    #     "incomplete_batches": batch_idx
                    # })
                    return  # Exit the training function
                # try:
                batch_x, batch_x_dt = batch_x.to(self.device), batch_x_dt.to(self.device)

                # Print SINDy parameters every N batches
                if batch_idx % 50 == 0:  # Adjust this value as needed
                    print(f"\nEpoch {epoch+1}, Batch {batch_idx}")
                    print("SINDy XI parameter matrix:")
                    print(self.model.XI.detach().cpu().numpy())
                    print("\nSINDy XI coefficient mask:")
                    print(self.model.XI_coefficient_mask.cpu().numpy())

                    # # If you're using wandb, you can log these values
                    # if wandb.run is not None:
                    #     wandb.log({
                    #         "SINDy_XI": wandb.Image(plt.imshow(self.model.XI.detach().cpu().numpy())),
                    #         "SINDy_XI_mask": wandb.Image(plt.imshow(self.model.XI_coefficient_mask.cpu().numpy())),
                    #     })
            
                
                self.optimizer.zero_grad()
                xh, dxh_dt, z, dz_dt, dz_dt_sindy = self.model(batch_x, batch_x_dt)
                loss, loss_dict = self.model.loss_function(batch_x,batch_x_dt, xh, dxh_dt, z, dz_dt, dz_dt_sindy)
                
                prediction_loss = loss
                
                prediction_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()        

                for key in loss_epoch.keys():
                    loss_epoch[key].append(loss_dict[key].item())
                    
                # except RuntimeError as e:
                #     print(f"Error in Epoch {e}")
                    # print(f"Error in Epoch {epoch+1}, Batch {batch_idx+1}")
                    # print(f"num_x shape: {batch_x.shape}")
                    # print(f"price_y shape: {batch_y.shape}")
                    # print(f"price_pred shape: {predictions.shape}")
                    # raise e
            
            for key in loss_epoch.keys():
                loss_list[key].append(sum(loss_epoch[key])/len(loss_epoch[key]))
                print(f"Epoch {epoch+1}/{self.model_hyperparameters['num_epochs']}, {key}: {loss_list[key][-1]}")
            
            del loss_epoch,loss_dict,xtilde, xtildedot, z, zdot, zdot_hat,loss
                    
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_x_dt in val_loader:
                    batch_x, batch_x_dt = batch_x.to(self.device), batch_x_dt.to(self.device)
                    xtilde, xtildedot, z, zdot, zdot_hat = self.model(batch_x, batch_x_dt)
                    loss, loss_dict = self.model.loss_function(batch_x,batch_x_dt, xtilde, xtildedot, zdot, zdot_hat)
                    val_loss += loss.item()
                    
                    
                    
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{self.model_hyperparameters['num_epochs']}, Val Loss: {val_loss:.3f}")
            # wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            # Early stopping
            if val_loss < best_val_loss:
                print("Saving model")
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_test.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break


        # if test_loader is None:
        #     print("No test data provided")
        #     print("Training completed")
        #     return
        
        # test_loss = 0.0
        # with torch.no_grad():
        #     for batch_x, batch_y in test_loader:
        #         batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        #         predictions = self.model(batch_x)
                
        #         prediction_loss = loss_function(predictions.squeeze()[:,-1,evaluation_columns], batch_y[:,-1,evaluation_columns])
                
        #         test_loss += prediction_loss.item()

        # test_loss /= len(test_loader)
        # print(f"Test Loss: {test_loss:.3f}")
        # wandb.log({"test_loss": test_loss})
        print("Training completed")
        return 
    
    def cleanup(self):
        del self.model
        del self.optimizer
        self.model = None
        self.optimizer = None
        torch.mps.empty_cache()
        gc.collect()