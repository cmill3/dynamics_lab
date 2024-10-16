import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.autograd as autograd
import wandb

def find_non_numeric(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if not is_supported_numeric(value):
                print(f"Non-numeric value found at index [{i}, {j}]: {value} (type: {type(value)})")
                return  # Stop after finding the first non-numeric value
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.device = None

    def setup(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("mps")

    def train(self,train_loader, val_loader, model_hyperparameters):
        # Define loss functions and optimizer
        loss_function = nn.HuberLoss()

        ## Enable anomaly detection
        # autograd.set_detect_anomaly(True)
        self.model.to(self.device)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        for epoch in range(model_hyperparameters["num_epochs"]):
            epoch_start = time.time()
            epoch_timeout = 1800
            self.model.train()
            train_loss = 0.0
            

            for batch_idx, (batch_x,batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_hyperparameters['num_epochs']}")):
                if batch_idx >= model_hyperparameters["batches_per_epoch"]:
                    break
                elif time.time() - epoch_start > epoch_timeout:
                    print(f"Epoch {epoch+1} timed out")
                    if model_hyperparameters['use_wandb']:
                        wandb.log({
                            "epoch_status": "timeout", 
                            "last_completed_epoch": epoch,
                            "incomplete_batches": batch_idx
                        })
                    return  # Exit the training function
                # try:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                predictions_loss = loss_function(predictions.squeeze()[:,-1,0], batch_y[:,-1,0])
                
                
                predictions_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()                
                train_loss += predictions_loss.item()
                    
                # except RuntimeError as e:
                #     print(f"Error in Epoch {e}")
                    # print(f"Error in Epoch {epoch+1}, Batch {batch_idx+1}")
                    # print(f"num_x shape: {batch_x.shape}")
                    # print(f"price_y shape: {batch_y.shape}")
                    # print(f"price_pred shape: {predictions.shape}")
                    # raise e
            
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {train_loss:.2f}")
                    
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_x)
                    
                    prediction_loss = loss_function(predictions.squeeze()[:,-1,0], batch_y[:,-1,0])
                    
                    val_loss += prediction_loss.item()
            
            val_loss /= len(val_loader)
            # Step the scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{model_hyperparameters['num_epochs']}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
            if model_hyperparameters['use_wandb']:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            # Early stopping
            if val_loss < best_val_loss:
                print("Saving model")
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        print("Training completed")
        return 
    
        
    # def train_categorical(self,train_alerts, val_alerts, model_hyperparameters, ts_data):
    #     # Define loss functions and optimizer
    #     loss_function = nn.MSELoss()

    #     ## Enable anomaly detection
    #     autograd.set_detect_anomaly(True)
    #     self.model.to(self.device)

    #     best_val_loss = float('inf')
    #     patience = 10
    #     patience_counter = 0

    #     scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    #     for epoch in range(model_hyperparameters["num_epochs"]):
    #         epoch_start = time.time()
    #         epoch_timeout = 1900
    #         self.model.train()
    #         train_loss = 0.0

    #         # Create random batches
    #         train_batches = create_random_batches(train_alerts, model_hyperparameters["batches_per_epoch"], model_hyperparameters["batch_size"])
            
    #         for batch_idx, (batch_alerts) in enumerate(tqdm(train_batches, desc=f"Epoch {epoch+1}/{model_hyperparameters['num_epochs']}")):
    #             if batch_idx >= model_hyperparameters["batches_per_epoch"]:
    #                 break
    #             elif time.time() - epoch_start > epoch_timeout:
    #                 print(f"Epoch {epoch+1} timed out")
    #                 # wandb.log({
    #                 #     "epoch_status": "timeout", 
    #                 #     "last_completed_epoch": epoch,
    #                 #     "incomplete_batches": batch_idx
    #                 # })
    #                 return  # Exit the training function
    #             try:
    #                 X_train, X_cat, Y_train = create_trend_sequences(ts_data, model_hyperparameters["context_length"], model_hyperparameters["prediction_horizon"], model_hyperparameters,batch_alerts)
    #                 try:
    #                     num_x = torch.FloatTensor(X_train)
    #                     cat_x = torch.LongTensor(X_cat)
    #                     batch_y = torch.FloatTensor(Y_train)
    #                 except Exception as e:
    #                     print("X_train")
    #                     continue
                        

    #                 num_x, batch_y, cat_x = num_x.to(self.device), batch_y.to(self.device), cat_x.to(self.device)                    
    #                 self.optimizer.zero_grad()

    #                 predictions = self.model(num_x, cat_x)
    #                 prediction_values, target_values = self.create_target_values(model_hyperparameters['mode'], predictions, batch_y)
    #                 predictions_loss = loss_function(prediction_values, target_values)
    #                 loss = predictions_loss
                    
                    
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #                 self.optimizer.step()                
    #                 train_loss += loss.item()
                    
    #             except RuntimeError as e:
    #                 print(f"Error in Epoch {epoch+1}, Batch {batch_idx+1}")
    #                 print(f"num_x shape: {batch_x.shape}")
    #                 print(f"price_y shape: {batch_y.shape}")
    #                 print(f"price_pred shape: {predictions.shape}")
    #                 raise e
            
    #         train_loss /= int(model_hyperparameters["batches_per_epoch"]*model_hyperparameters["batch_size"])
    #         print(f"Epoch {epoch+1}, Loss: {train_loss}")

    #         del X_train, X_cat, Y_train
    #         del num_x, cat_x, batch_y
                    
    #         # Validation
    #         self.model.eval()
    #         val_loss = 0.0
    #         num_batches = len(val_alerts) / model_hyperparameters["batch_size"]
    #         val_batches = create_random_batches(train_alerts, num_batches, model_hyperparameters["batch_size"])
            
    #         with torch.no_grad():
    #             for batch_idx, (batch_alerts) in enumerate(tqdm(val_batches, desc=f"Val Epoch {epoch+1}/{len(val_batches)}")):
    #                 X_train, X_cat, Y_train = create_trend_sequences(ts_data, model_hyperparameters["context_length"], model_hyperparameters["prediction_horizon"], model_hyperparameters,batch_alerts)
    #                 try:
    #                     num_x = torch.FloatTensor(X_train)
    #                     cat_x = torch.LongTensor(X_cat)
    #                     batch_y = torch.FloatTensor(Y_train)
    #                 except:
    #                     print("X_val")
    #                     continue

    #                 batch_x, batch_y, cat_x = num_x.to(self.device), batch_y.to(self.device), cat_x.to(self.device)
    #                 predictions = self.model(batch_x, cat_x)
                    
    #                 prediction_values, target_values = self.create_target_values(model_hyperparameters['mode'], predictions, batch_y)
    #                 predictions_loss = loss_function(prediction_values, target_values)
                    
    #                 val_loss += predictions_loss.item()
        
    #         val_loss /= len(val_alerts)
    #         # Step the scheduler
    #         scheduler.step(val_loss)

    #         del X_train, X_cat, Y_train
    #         del num_x, cat_x, batch_y
            
    #         print(f"Epoch {epoch+1}/{model_hyperparameters['num_epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    #         # wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    #         # Early stopping
    #         if val_loss < best_val_loss:
    #             print("Saving model")
    #             best_val_loss = val_loss
    #             patience_counter = 0
    #             torch.save(self.model.state_dict(), 'best_model.pth')
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= patience:
    #                 print("Early stopping triggered")
    #                 break

    #         if (best_val_loss > 0.5) and (epoch > 10):
    #             print("Early stopping triggered")
    #         # wandb.log({"val_loss": best_val_loss})
    #         print("Training completed")
    #     return 

    # def create_target_values(self, mode, predictions, targets):
    #     if mode == 'max_estimation':
    #         prediction_values = predictions[:, :, 1]
    #         target_values = targets[:, :, 1]
    #         target_value = torch.max(target_values, dim=1).values
    #         prediction_value = torch.max(prediction_values, dim=1).values
    #     elif mode == 'min_estimation':
    #         prediction_values = predictions[:, :, 2]
    #         target_values = targets[:, :, 2]
    #         target_value = torch.max(target_values, dim=1).values
    #         prediction_value = torch.max(prediction_values, dim=1).values

    #     return prediction_value, target_value
        
            


    def cleanup(self):
        del self.model
        del self.optimizer
        self.model = None
        self.optimizer = None
        torch.mps.empty_cache()
        gc.collect()

