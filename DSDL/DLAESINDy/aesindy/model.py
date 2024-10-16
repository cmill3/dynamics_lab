import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pysindy as ps
import keras
import wandb
from tensorflow.keras.callbacks import Callback
from pysindy.feature_library import FourierLibrary, PolynomialLibrary
from pysindy.feature_library import GeneralizedLibrary
from itertools import combinations_with_replacement

    
class SindyCall(tf.keras.callbacks.Callback):
    def __init__(self, threshold, update_freq, x, poly_order=2, include_fourier=False, n_frequencies=1, input_dim=128):
        super(SindyCall, self).__init__()
        self.threshold = threshold 
        self.update_freq = update_freq
        self.x = x
        self.poly_order = poly_order
        self.include_fourier = include_fourier
        self.n_frequencies = n_frequencies
        self.input_dim = input_dim
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_freq == 0 and epoch > 1:
            print()
            print('--- Running Sindy ---')
            x_in = self.x[:, :self.input_dim]
            z = self.model.encoder(x_in)
            z_latent = z.numpy()
            
            # Create PySINDy libraries
            poly_library = PolynomialLibrary(
                degree=self.poly_order,
                include_interaction=True
            )

            if self.include_fourier:
                fourier_library = FourierLibrary(
                    n_frequencies=self.n_frequencies
                )
                library = GeneralizedLibrary(
                    [poly_library, fourier_library]
                )
            else:
                library = poly_library

            opt = ps.optimizers.STLSQ(threshold=self.threshold)
            sindy_model = ps.SINDy(feature_library=library, optimizer=opt)
            time = np.linspace(0, self.model.params['dt']*z_latent.shape[0], z_latent.shape[0], endpoint=False)
            sindy_model.fit(z_latent, t=time)
            sindy_model.print()
            print(sindy_model.coefficients().T)
            
            sindy_weights = sindy_model.coefficients().T  
            sindy_mask = sindy_model.coefficients().T > 1e-5
            layer_weights = [sindy_mask, sindy_weights]
            
            self.model.sindy.set_weights(layer_weights)
                
        
        
# This callback is used to update mask, i.e. apply recursive feature elimination in Sindy at the beginning of each epoch
class RfeUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, rfe_frequency=0):
        super(RfeUpdateCallback, self).__init__()
        self.rfe_frequency = rfe_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        # if epoch % self.model.print_frequency == 0:
        #     # print()
        #     # print('--- Sindy Coefficients ---')
        #     # print(self.model.sindy.coefficients.numpy())
        if epoch % self.model.rfe_frequency == 0:
            self.model.sindy.update_mask()
        
    def on_train_begin(self, logs=None):
        print('--- Initial Sindy Coefficients ---')
        print(self.model.sindy.coefficients.numpy())



class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log({"epoch": epoch})


# class NaNCallback(Callback):
#     def on_epoch_end(self, logs=None):
#         if np.isnan(logs.get('loss')):
#             print('NaN loss detected, terminating training')
#             self.model.stop_training = True
#             wandb.run.summary['terminated_on_nan'] = True


# Try to cast as a layer.Layer object
# Add sindy_library_tf function to class
class Sindy(layers.Layer): 
    def __init__(self, library_dim, 
                 state_dim, 
                 poly_order, 
                 model='lorenz', 
                 initializer='constant', 
                 actual_coefs=None, 
                 rfe_threshold=None, 
                 include_fourier=False, 
                 exact_features=False, 
                 fix_coefs=False, 
                 sindy_pert=0.0, 
                 ode_net=False,
                 ode_net_widths=[1.5, 2.0],
                 init_scale=7.0,
                n_frequencies=1,
                 **kwargs):
        super(Sindy, self).__init__(**kwargs)
        
        self.library_dim = library_dim
        self.state_dim = state_dim
        self.poly_order = poly_order
        self.include_fourier = include_fourier
        self.n_frequencies = n_frequencies
        self.rfe_threshold = rfe_threshold
        self.exact_features = exact_features
        self.actual_coefs = actual_coefs
        self.fix_coefs = fix_coefs
        self.sindy_pert = sindy_pert 
        self.model = model
        self.init_scale = init_scale
        
        self.ode_net = ode_net 
        self.ode_net_widths = ode_net_widths
        self.l2 = 1e-6
        self.l1 = 0.0 

        ## INITIALIZE COEFFICIENTS

        if type(initializer) == np.ndarray:
            self.coefficients_mask = tf.Variable(initial_value=np.abs(initializer)>1e-10, dtype=tf.float32)
            self.coefficients = tf.Variable(initial_value= initializer, name='sindy_coeffs', dtype=tf.float32)
        else:
            if initializer == 'true':
                self.coefficients_mask = tf.Variable(initial_value=np.abs(actual_coefs)>1e-10, dtype=tf.float32)
                self.coefficients = tf.Variable(initial_value=actual_coefs + sindy_pert*(np.random.random(actual_coefs.shape)-0.5), name='sindy_coeffs', dtype=tf.float32)
            else:
                if initializer == 'variance_scaling':
                    init = tf.keras.initializers.VarianceScaling(scale=10, mode='fan_in', distribution='uniform')
                elif initializer == 'constant':
                    init = tf.constant_initializer(0.0)
                elif type(initializer) != str:
                    init = tf.constant_initializer(initializer)
                elif initializer == 'random_normal':
                    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=init_scale)
                else:
                    raise Exception("initializer string doesn't exist")
                
                self.coefficients = self.add_weight(
                    shape=(self.library_dim, self.state_dim),
                    initializer='random_normal',
                    trainable=True,
                    name='coefficients'
                )
                self.coefficients_mask = self.add_weight(
                    shape=(self.library_dim, self.state_dim),
                    initializer='ones',
                    trainable=False,
                    name='coefficients_mask'
                )



        if self.fix_coefs:
            self.coefficients = tf.Variable(initial_value=actual_coefs, name='sindy_coeffs', trainable=False, dtype=tf.float32)
            
        ## ODE NET
        if self.ode_net:
            self.net_model = self.make_theta_network(self.library_dim, self.ode_net_widths)


    def make_theta_network(self, output_dim, widths):
        out_activation = 'linear'
        name = 'net_dictionary'
        initializer= tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform")
        model = tf.keras.Sequential()
        for i, w in enumerate(widths):
            model.add(tf.keras.layers.Dense(w, activation='elu', kernel_initializer=initializer, 
               name=name+'_'+str(i), use_bias=True, activity_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)))
        model.add(tf.keras.layers.Dense(output_dim, activation=out_activation, 
               kernel_initializer=initializer, name=name+'_'+'out', use_bias=True))
        return model
    
    def call(self, z):
        dz_dt = tf.matmul(self.theta(z), self.coefficients)
        return dz_dt
    
    @tf.function
    def theta(self, z):
        if self.ode_net:
            return self.net_model(z)
        else:
            return self.sindy_library_tf(z, self.state_dim, self.poly_order, self.include_fourier, self.n_frequencies)
    
    def update_mask(self):
        if self.rfe_threshold is not None:
            print()
            print('--- Running RFE ---')
            self.coefficients_mask.assign(tf.cast( tf.abs(self.coefficients) > self.rfe_threshold ,tf.float32) )
            self.coefficients.assign(tf.multiply(self.coefficients_mask, self.coefficients))
            # print(self.coefficients_mask.numpy())
            # print(self.coefficients.numpy())

    
    @tf.function 
    def sindy_library_tf(self, z, latent_dim, poly_order, include_fourier=False,n_frequencies=False):
        # Can make more compact
        library = [tf.ones(tf.shape(z)[0])]
        for i in range(latent_dim):
            library.append(z[:,i])

        if poly_order > 1:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    library.append(tf.multiply(z[:,i], z[:,j]))

        if poly_order > 2:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    for k in range(j,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k])

        if poly_order > 3:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    for k in range(j,latent_dim):
                        for p in range(k,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

        if poly_order > 4:
            for i in range(latent_dim):
                for j in range(i,latent_dim):
                    for k in range(j,latent_dim):
                        for p in range(k,latent_dim):
                            for q in range(p,latent_dim):
                                library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

        # Add Fourier terms
        if include_fourier:
            for i in range(latent_dim):
                for k in range(1, n_frequencies + 1):
                    library.append(tf.sin(k * z[:,i]))
                    library.append(tf.cos(k * z[:,i]))

        return tf.stack(library, axis=1)    


    #######################################################

# Put in class (?)
total_met = tf.keras.metrics.Mean(name="total_loss")
rec_met = tf.keras.metrics.Mean(name="rec_loss")
sindy_z_met = tf.keras.metrics.Mean(name="sindy_z_loss")
sindy_x_met = tf.keras.metrics.Mean(name="sindy_x_loss")
integral_met = tf.keras.metrics.Mean(name="integral_loss")
x0_met = tf.keras.metrics.Mean(name='x0_loss')
l1_met = tf.keras.metrics.Mean(name='l1_loss')
prediction_met = tf.keras.metrics.Mean(name='prediction_loss')

@keras.saving.register_keras_serializable(package="SindyAutoencoder")
class SindyAutoencoder(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super(SindyAutoencoder, self).__init__(**kwargs)
        self.params = params
        self.latent_dim = params['latent_dim']
        self.input_dim = params['input_dim'] * params['n_time_series']
        self.widths = params['widths']
        self.activation = params['activation']
        self.library_dim = params['library_dim']
        self.poly_order = params['poly_order']
        self.include_fourier = params['include_fourier']
        self.n_frequencies = params['n_frequencies']
        self.initializer = params['coefficient_initialization'] # fix That's sindy's!
        self.epochs = params['max_epochs'] # fix That's sindy's!
        self.rfe_threshold = params['coefficient_threshold']
        self.rfe_frequency = params['threshold_frequency']
        self.print_frequency = params['print_frequency']
        self.sindy_pert = params['sindy_pert']
        self.fixed_coefficient_mask = None
        self.actual_coefs = params['actual_coefficients']
        self.use_bias = params['use_bias']
        self.l2 = params['loss_weight_layer_l2']
        self.l1 = params['loss_weight_layer_l1']
        self.sparse_weights = params['sparse_weighting']
        self.trainable_auto = params['trainable_auto']
        self.variable_weights = tf.constant(value=params['variable_weights'], dtype=tf.float32)
        if params['sparse_weighting'] is not None:
            self.sparse_weights = tf.constant(value=params['sparse_weighting'], dtype=tf.float32)
        if params['fixed_coefficient_mask']:
            self.fixed_coefficient_mask = tf.constant(value=np.abs(self.actual_coefs)>1e-10, dtype=tf.float32) 

        self.time = tf.constant(value=np.linspace(0.0, params['dt']*params['input_dim'], params['input_dim'], endpoint=False))
        self.dt = tf.constant(value=params['dt'], dtype=tf.float32)
        
        self.encoder = self.make_network(self.input_dim, self.latent_dim, self.widths, name='encoder')
        self.decoder = self.make_network(self.latent_dim, self.input_dim, self.widths[::-1], name='decoder')
        if not self.trainable_auto:
            self.encoder._trainable = False
            self.decoder._trainable = False
        self.sindy = Sindy(self.library_dim, self.latent_dim, self.poly_order, model=params['model'], 
                           initializer=self.initializer, actual_coefs=self.actual_coefs, rfe_threshold=self.rfe_threshold, 
                           include_fourier=self.include_fourier, exact_features=params['exact_features'], fix_coefs=params['fix_coefs'], sindy_pert=self.sindy_pert, 
                           ode_net=params['ode_net'], ode_net_widths=params['ode_net_widths'], init_scale=params['sindy_init_scale'], n_frequencies=self.n_frequencies)

    
    def make_network(self, input_dim, output_dim, widths, name):
        out_activation = 'linear'
        initializer= tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform")
        model = tf.keras.Sequential()
        for i, w in enumerate(widths):
            if i ==0:
                use_bias = self.use_bias
            else:
                use_bias = True 
            model.add(tf.keras.layers.Dense(w, activation=self.activation, kernel_initializer=initializer, name=name+'_'+str(i), use_bias=use_bias, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)))
        model.add(tf.keras.layers.Dense(output_dim, activation=out_activation, kernel_initializer=initializer, name=name+'_'+'out', use_bias=self.use_bias))
        return model
                
    def call(self, datain):
        x = datain[0][:, :self.params['input_dim']]
        return self.decoder(self.encoder(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({"params": self.params})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @tf.function
    def predict_future(self, x_initial, n_steps):
        z_current = self.encoder(x_initial)
        x_future = tf.TensorArray(tf.float32, size=n_steps+1)
        x_future = x_future.write(0, x_initial)
        
        for i in tf.range(n_steps):
            dz_dt = self.sindy(z_current)
            z_current = z_current + dz_dt * self.dt
            x_current = self.decoder(z_current)
            x_future = x_future.write(i+1, x_current)
        
        x_future = x_future.stack()
        x_future = tf.transpose(x_future, [1, 0, 2])  # Rearrange to [batch, time, features]
        x_future = x_future[:, 1:, -1:] # Remove the last feature, which is the future feature
         # Ensure the output shape matches x_future
        if x_future.shape.ndims == 3 and x_future.shape[-1] == 1:
            x_future = tf.squeeze(x_future, axis=-1)
        return x_future


    @tf.function
    def predict_future(self, x_initial, n_steps):
        z_initial = self.encoder(x_initial)
        z_future = tf.TensorArray(tf.float32, size=n_steps+1)
        z_future = z_future.write(0, z_initial)
        
        for i in tf.range(n_steps):
            current_z = z_future.read(i)
            dz_dt = self.sindy(current_z)
            z_next = current_z + dz_dt * self.dt
            z_future = z_future.write(i+1, z_next)
        
        z_future = z_future.stack()
        z_future = tf.transpose(z_future, [1, 0, 2])  # Rearrange to [batch, time, features]
        x_future = self.decoder(z_future)
        
        return x_future

    
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        x = inputs[0][:, :self.params['input_dim']]
        x_future = inputs[0][:, self.params['input_dim']:]
        dx_dt = tf.expand_dims(inputs[1][:,:self.params['input_dim']], 2)
        x_out = outputs[0][:, :self.params['input_dim']]
        dx_dt_out =  tf.expand_dims(outputs[1][:,:self.params['input_dim']], 2)


        with tf.GradientTape() as tape:
            loss, losses = self.get_loss(x, dx_dt, x_out, dx_dt_out, x_future)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        ## Keep track and update losses
        self.update_losses(loss, losses)
        metrics =  {m.name: m.result() for m in self.metrics}
        return metrics

    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        x = inputs[0][:, :self.params['input_dim']]
        x_future = inputs[0][:, self.params['input_dim']:]
        dx_dt = tf.expand_dims(inputs[1][:,:self.params['input_dim']], 2)
        x_out = outputs[0][:, :self.params['input_dim']]
        dx_dt_out =  tf.expand_dims(outputs[1][:,:self.params['input_dim']], 2)
        
        loss, losses = self.get_loss(x, dx_dt, x_out, dx_dt_out, x_future)
        
        ## Keep track and update losses
        self.update_losses(loss, losses)
        metrics =  {m.name: m.result() for m in self.metrics}
        return metrics

    

    @tf.function
    def get_loss(self, x, dx_dt, x_out, dx_dt_out, x_future):
        losses = {}
        loss = 0
        if self.params['loss_weight_sindy_z'] > 0.0:
            with tf.GradientTape() as t1:
                t1.watch(x)
                z = self.encoder(x, training=self.trainable_auto)
            dz_dx = t1.batch_jacobian(z, x)
            dz_dt = tf.matmul( dz_dx, dx_dt )
        else:
            z = self.encoder(x, training=self.trainable_auto)

        if self.params['loss_weight_sindy_x'] > 0.0:
            with tf.GradientTape() as t2:
                t2.watch(z)
                xh = self.decoder(z, training=self.trainable_auto)
            dxh_dz = t2.batch_jacobian(xh, z)
            dz_dt_sindy = tf.expand_dims(self.sindy(z), 2)
            dxh_dt = tf.matmul( dxh_dz, dz_dt_sindy )
        else:
            xh = self.decoder(z, training=self.trainable_auto) 

        # SINDy consistency loss
        if self.params['loss_weight_integral'] > 0.0:
            sol = z 
            loss_int = tf.square(sol[:, 0] - x[:, 0]) 
            total_steps = len(self.time)
            for i in range(1, len(self.time)):
                k1 = self.sindy(sol)
                k2 = self.sindy(sol + self.dt/2 * k1)
                k3 = self.sindy(sol + self.dt/2 * k2)
                k4 = self.sindy(sol + self.dt * k3)
                sol = sol + 1/6 * self.dt * (k1 + 2*k2 + 2*k3 + k4) 
                # To avoid nans - tf.where(tf.is_nan()) not compatible with gradients 
                sol = tf.where(tf.abs(sol) > 500.00, 500.0, sol) 
                loss_int += tf.square(sol[:, 0] - x[:, i])
            loss += self.params['loss_weight_integral'] * loss_int / total_steps 
            losses['integral'] = loss_int


        if self.params['loss_weight_sindy_x'] > 0.0:
            # loss_dx = tf.reduce_mean( tf.square(dxh_dt - dx_dt_out) ) 
            loss_dx = tf.reduce_mean(
                tf.multiply(
                    self.variable_weights,
                    tf.reduce_mean(tf.square(dxh_dt - dx_dt_out), axis=0)
                )
            )
            loss += self.params['loss_weight_sindy_x'] * loss_dx
            losses['sindy_x'] = loss_dx

        if self.params['loss_weight_sindy_z'] > 0.0:
            loss_dz = tf.reduce_mean( tf.square(dz_dt - dz_dt_sindy) )
            loss += self.params['loss_weight_sindy_z'] * loss_dz
            losses['sindy_z'] = loss_dz

        if self.params['loss_weight_x0'] > 0.0:
            loss_x0 = tf.reduce_mean( tf.square(z[:, 0] - x[:, 0]) )
            loss += self.params['loss_weight_x0'] * loss_x0
            losses['x0'] = loss_x0

        if self.params['loss_weight_prediction'] > 0.0:
            future_steps = self.params['future_steps']
            prediction = self.predict_future(x, future_steps)
            loss_pred = tf.reduce_mean( tf.square(x_future - prediction) )
            loss += self.params['loss_weight_prediction'] * loss_pred
            losses['prediction'] = loss_pred

        loss_rec = tf.reduce_mean( tf.square(xh - x_out) )
        # loss_rec = tf.reduce_mean(
        #     tf.multiply(
        #         self.variable_weights,
        #         tf.reduce_mean(tf.square(xh - x_out), axis=0)
        #     )
        # )

        # Add future prediction loss
        x_pred_future = self.predict_future(x[:, 0], x_future.shape[1])
        loss_future = tf.reduce_mean(tf.square(x_pred_future - x_future))
        loss += self.params['loss_weight_prediction'] * loss_future
        losses['prediction'] = loss_future
        
        if self.sparse_weights is not None:
            loss_l1 = tf.reduce_mean(tf.abs(tf.multiply(self.sparse_weights, self.sindy.coefficients) ) )
        else:
            loss_l1 = tf.reduce_mean(tf.abs(self.sindy.coefficients) )
            
        loss += self.params['loss_weight_rec'] * loss_rec \
              + self.params['loss_weight_sindy_regularization'] * loss_l1

        if self.fixed_coefficient_mask is not None:
            self.sindy.coefficients.assign( tf.multiply(self.sindy.coefficients, self.fixed_coefficient_mask) )

        losses['rec'] = loss_rec
        losses['l1'] = loss_l1

        return loss, losses
    
    @tf.function
    def update_losses(self, loss, losses):
        total_met.update_state(loss)
        rec_met.update_state(losses['rec'])
        if self.params['loss_weight_sindy_z'] > 0:
            sindy_z_met.update_state(losses['sindy_z'])
        if self.params['loss_weight_sindy_x'] > 0:
            sindy_x_met.update_state(losses['sindy_x'])
        if self.params['loss_weight_integral'] > 0:
            integral_met.update_state(losses['integral'])
        if self.params['loss_weight_x0'] > 0:
            x0_met.update_state(losses['x0'])
        if self.params['loss_weight_sindy_regularization'] > 0:
            l1_met.update_state(losses['l1'])
        if self.params['loss_weight_prediction'] > 0:
            prediction_met.update_state(losses['prediction'])

    # Check if needed
    @property
    def metrics(self):
        m = [total_met, rec_met, prediction_met]
        if self.params['loss_weight_sindy_z'] > 0.0:
            m.append(sindy_z_met)
        if self.params['loss_weight_sindy_x'] > 0.0:
            m.append(sindy_x_met)
        if self.params['loss_weight_integral'] > 0.0:
            m.append(integral_met)
        if self.params['loss_weight_x0'] > 0.0:
            m.append(x0_met)
        if self.params['loss_weight_sindy_regularization'] > 0.0:
            m.append(l1_met)
        if self.params['loss_weight_prediction'] > 0.0:
            m.append(prediction_met)
        return m 

