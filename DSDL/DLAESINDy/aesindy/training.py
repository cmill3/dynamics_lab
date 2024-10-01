import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import datetime
import numpy as np
import pandas as pd
import pickle5 as pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import tensorflow as tf
import pdb
from sklearn.preprocessing import StandardScaler
from .sindy_utils import library_size, sindy_library
from .model import SindyAutoencoder, PreSVDSindyAutoencoder, RfeUpdateCallback, SindyCall


class TrainModel:
    def __init__(self, data, params):
        self.data = data
        self.params = self.fix_params(params)
        self.model = self.get_model()
        self.savename = self.get_name()
        self.history = None
        
    def get_name(self, include_date=True):
        pre = 'results'
        post = self.params['model']+'_'+self.params['case']
        if include_date:
            name = pre + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '_' + post
        else:
            name = pre + '_' + post
        return name 
    
    # def fix_params(self, params):
    #     input_dim = params['input_dim']
    #     if params['svd_dim'] is not None:
    #         print(params['svd_dim'])
    #         print('Running SVD decomposition...')
    #         input_dim = params['svd_dim']
    #         reduced_dim = int(params['svd_dim'])
            
    #         # Transpose the input data to have shape (29677, 128)
    #         X_transposed = self.data['x'].T
            
    #         # Perform SVD on the transposed data
    #         U, s, VT = np.linalg.svd(X_transposed, full_matrices=False)
            
    #         # Take only the first 'reduced_dim' components
    #         U_reduced = U[:, :reduced_dim]
    #         s_reduced = s[:reduced_dim]
            
    #         # Compute the reduced representation
    #         v = np.dot(U_reduced, np.diag(s_reduced))  # Shape will be (29677, 6)
            
    #         if params['scale'] == True:
    #             scaler = StandardScaler()
    #             v = scaler.fit_transform(v)
            
    #         self.data['xorig'] = self.data['x']
    #         self.data['x'] = v  # Shape is (29677, 6)

    #         print(self.data['x'].shape)

    #         #  Assumes 1 IC 
    #         self.data['dx'] = np.gradient(v, params['dt'], axis=0)
    #         print('SVD Done!')
        
    #     print(self.data.keys())
    #     print(self.data['t'].shape)
    #     print(self.data['x'].shape)
    #     print(self.data['dx'].shape)
    #     print(self.data['z'].shape)
    #     print(self.data['dz'].shape)
    #     # print(self.data['sindy_coefficients'].shape)
    #     print(self.data['xorig'].shape)

        
    def fix_params(self, params):
        input_dim = params['input_dim']
        if params['svd_dim'] is not None:
            print(params['svd_dim'])
            print('Running SVD decomposition...')
            input_dim = params['svd_dim']
            reduced_dim = int( params['svd_dim'] )
            U, s, VT = np.linalg.svd(self.data['x'], full_matrices=False)
            v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
            if params['scale'] == True:
                scaler = StandardScaler()
                v = scaler.fit_transform(v)
            self.data['xorig'] = self.data['x']
            self.data['x'] = v.T

            print(self.data['x'].shape)

            
            #  Assumes 1 IC 
            self.data['dx'] = np.gradient(v, params['dt'], axis=0).T
            print('SVD Done!')
            
        print(self.data['x'].shape)
        print(self.data['dx'].shape)
        print(self.data['z'].shape)
        print(self.data['dz'].shape)
        # print(self.data['xorig'].shape)
        params['widths'] = [int(i*input_dim) for i in params['widths_ratios']]
        
        ## Constraining features according to model/case
        if params['exact_features']:
            if params['model'] == 'lorenz':
                params['library_dim'] = 5
                self.data.sindy_coefficients = self.data.sindy_coefficients[np.array([1, 2, 3, 5, 6]), :]
            elif params['model'] == 'rossler':
                params['library_dim'] = 5
                self.data.sindy_coefficients = self.data.sindy_coefficients[np.array([0, 1, 2, 3, 6]), :]
            elif params['model'] == 'predprey':
                params['library_dim'] = 3
                self.data.sindy_coefficients = self.data.sindy_coefficients[np.array([1, 2, 4]), :]
        else:
            params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)
            
        if hasattr(self.data, 'sindy_coefficients') and self.data.sindy_coefficients is not None:
            params['actual_coefficients'] = self.data.sindy_coefficients
        else:
            params['actual_coefficients'] = None

        if 'sparse_weighting' in params:
            if params['sparse_weighting'] is not None:
                a, sparse_weights = sindy_library(self.data.z[:100, :], params['poly_order'], include_sparse_weighting=True)
                params['sparse_weighting'] = sparse_weights
        
        return params

    def get_data(self):
        # Split into train and test sets
        train_x, test_x = train_test_split(self.data['x'].T, train_size=self.params['train_ratio'], shuffle=False)
        train_dx, test_dx = train_test_split(self.data['dx'].T, train_size=self.params['train_ratio'], shuffle=False)
        train_data = [train_x, train_dx]  
        test_data = [test_x, test_dx]  
        if self.params['svd_dim'] is not None:
            train_xorig, test_xorig = train_test_split(self.data['xorig'].T, train_size=self.params['train_ratio'], shuffle=False)
            train_data = [train_xorig] + train_data
            test_data = [test_xorig] + test_data 
            
        return train_data, test_data

    # def get_data(self):
    #     # Split into train and test sets
    #     train_x, test_x = train_test_split(self.data['x'].T, train_size=self.params['train_ratio'], shuffle=False)
    #     train_dx, test_dx = train_test_split(self.data['dx'].T, train_size=self.params['train_ratio'], shuffle=False)
        
    #     # Ensure all data has the same shape
    #     train_x = np.expand_dims(train_x, axis=-1)
    #     test_x = np.expand_dims(test_x, axis=-1)
    #     train_dx = np.expand_dims(train_dx, axis=-1)
    #     test_dx = np.expand_dims(test_dx, axis=-1)
        
    #     train_data = np.concatenate([train_x, train_dx], axis=-1)
    #     test_data = np.concatenate([test_x, test_dx], axis=-1)
        
    #     if self.params['svd_dim'] is not None:
    #         train_xorig, test_xorig = train_test_split(self.data['xorig'], train_size=self.params['train_ratio'], shuffle=False)
    #         train_xorig = np.expand_dims(train_xorig, axis=-1)
    #         test_xorig = np.expand_dims(test_xorig, axis=-1)
    #         train_data = np.concatenate([train_xorig, train_data], axis=-1)
    #         test_data = np.concatenate([test_xorig, test_data], axis=-1)
    
    #     return train_data, test_data

    def get_model(self):
        if self.params['svd_dim'] is None:
            model = SindyAutoencoder(self.params)
        else:
            model = PreSVDSindyAutoencoder(self.params)
        return model

    def fit(self):
        train_data, test_data = self.get_data()
        self.save_params()
        print(self.savename)
        
        # Create directory and file name
        os.makedirs(os.path.join(self.params['data_path'], self.savename), exist_ok=True)
        os.makedirs(os.path.join(self.params['data_path'], self.savename, 'checkpoints'), exist_ok=True)
        
        # Build model and fit
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='mse')

        callback_list = get_callbacks(self.params, self.savename, x=test_data[1])
        print('Fitting model..')
        for x in train_data:
            print(x.shape)
        self.history = self.model.fit(
                x=train_data, y=train_data, 
                batch_size=self.params['batch_size'],
                epochs=self.params['max_epochs'], 
                validation_data=(test_data, test_data),
                callbacks=callback_list,
                shuffle=True)
        
        if self.params['case'] != 'lockunlock':
            prediction = self.model.predict(test_data)
            self.save_results(self.model)
            
        else: # Used to make SINDy coefficients trainable 
            self.params['fix_coefs'] = False
            self.model_unlock = self.get_model()
            self.model_unlock.predict(test_data) # For building model, required for transfer 
            self.model_unlock.set_weights(self.model.get_weights()) # Transfer weights 
            self.model_unlock.compile(optimizer=optimizer, loss='mse')
            self.history = self.model_unlock.fit(
                    x=train_data, y=train_data, 
                    batch_size=self.params['batch_size'],
                    epochs=self.params['max_epochs'], 
                    validation_data=(test_data, test_data),
                    callbacks=callback_list,
                    shuffle=True)
            prediction = self.model_unlock.predict(test_data)
            self.save_results(self.model_unlock)
        return prediction
        
    def get_results(self, model):
        results_dict = {}
        results_dict['losses'] = self.history.history 
        results_dict['sindy_coefficients'] = model.sindy.coefficients.numpy()
        
        return results_dict

    def save_params(self):
        saving_params = self.params
            
        # Save parameters
        # Create a DataFrame
        os.makedirs(os.path.join(self.params['data_path'], self.savename), exist_ok=True)
        df = pd.DataFrame([saving_params])
        df.to_pickle(os.path.join(saving_params['data_path'], self.savename ,'model_params.pkl'))
            
    def save_results(self, model):
        print('Saving results..')
        print(self.get_results(model))
        df = pd.DataFrame([self.get_results(model)])
        df.to_pickle(os.path.join(self.params['data_path'], self.savename,'model_results.pkl'))

        # Save model
        # pdb.set_trace()
        model.save(os.path.join(self.params['data_path'], self.savename,'model.keras'))

#########################################################


#########################################################
#########################################################

def get_callbacks(params, savename, x=None, t=None):
    callback_list = []
    
    ## Tensorboad Saving callback - Good for analyzing results
    def get_run_logdir(current_dir=os.curdir):
        root_logdir = os.path.join(current_dir, 'my_logs')
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S_")
        return os.path.join(root_logdir, run_id)

    # Update coefficient_mask callback
    if params['coefficient_threshold'] is not None:
        callback_list.append(RfeUpdateCallback(rfe_frequency=params['threshold_frequency']))
    
    # Early stopping in when training stops improving
    if params['patience'] is not None:
        callback_list.append(tf.keras.callbacks.EarlyStopping(patience=params['patience'], monitor='val_rec_loss', mode='min', restore_best_weights=True))


    # Learning rate scheduler - Decrease learning rate exponentially (include in callback if needed)
    if params['learning_rate_sched']:
        def exponential_decay(lr0, s):
            def exponential_decay_fn(epoch):
                return lr0 * 0.1**(epoch/s)
            return exponential_decay_fn
        exponential_decay_fn = exponential_decay(lr0=params['learning_rate'], s=params['max_epochs'])
        callback_list.append( tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn) )
        
    if params['save_checkpoints']:
        checkpoint_path = os.path.join(params['data_path'], savename, 'checkpoints', 'cp-{epoch:04d}.weights.h5')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_path, 
                        verbose=1, 
                        save_weights_only=True,
                        save_freq=params['save_freq'] * int(params['tend']/params['dt']*params['n_ics']/ \
                                            params['batch_size'] * params['train_ratio']))
        
        callback_list.append(cp_callback)
        
    if params['use_sindycall']:
        print('generating data for sindycall')
        params2 = params.copy()
        params2['tend'] = 200
        params2['n_ics'] = 1

        # Change and NOT TESTED 
        data2 = self.data.copy()
        data2.run_sim(params2['n_ics'], params2['tend'], params2['dt'])

        print('Done..')
        x = data2.x
        t = data2.t[:data_test.x.shape[0]]
        callback_list.append(SindyCall(threshold=params2['sindy_threshold'], update_freq=params2['sindycall_freq'], x=x, t=t))
        
    return callback_list