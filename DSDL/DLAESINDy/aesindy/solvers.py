import numpy as np
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from .dynamical_models import get_model
from .helper_functions import get_hankel
from tqdm import tqdm
import pdb



class DatasetConstructorSynth:
    def __init__(self, 
        model='lorenz',
        args=None, 
        noise=0.0, 
        input_dim=128,
        normalization=None):

        self.model = model
        self.args = args
        self.noise = noise
        self.input_dim = input_dim
        self.normalization = None 

    def solve_ivp(self, f, z0, time):
        """ Scipy ODE solver, returns z and dz/dt """
        z = odeint(f, z0, time)
        dz = np.array([f(z[i], time[i]) for i in range(len(time))])
        return z, dz
    
    def run_sim(self, n_ics, tend, dt, z0_stat=None):
        """ Runs solver over multiple initial conditions and builds Hankel matrix """

        f, Xi, model_dim, z0_mean_sug, z0_std_sug = get_model(self.model, self.args, self.normalization)
        self.normalization = self.normalization if self.normalization is not None else np.ones((model_dim,))
        if z0_stat is None:
            z0_mean = z0_mean_sug
            z0_std = z0_std_sug
        else:
            z0_mean, z0_std = z0_stat

        time = np.arange(0, tend, dt)
        z0_mean = np.array(z0_mean) 
        z0_std =  np.array(z0_std) 
        z0 = z0_std*(np.random.rand(n_ics, model_dim)-.5) + z0_mean 

        delays = len(time) - self.input_dim
        z_full, dz_full, H, dH = [], [], [], []
        print("generating solutions..")
        for i in tqdm(range(n_ics)):
            z, dz = self.solve_ivp(f, z0[i, :], time)
            z *= self.normalization
            dz *= self.normalization

            # Build true solution (z) and hankel matrices
            z_full.append( z[:-self.input_dim, :] )
            dz_full.append( dz[:-self.input_dim, :] )
            x = z[:, 0] + self.noise * np.random.randn(len(time),) # Assumes first dim measurement
            dx = dz[:, 0] + self.noise * np.random.randn(len(time),) # Assumes first dim measurement
            H.append( get_hankel(x, self.input_dim, delays) )
            dH.append( get_hankel(dx, self.input_dim, delays) )
        
        self.z = np.concatenate(z_full, axis=0)
        self.dz = np.concatenate(dz_full, axis=0)
        self.x = np.concatenate(H, axis=1) 
        self.dx = np.concatenate(dH, axis=1) 
        self.t = time
        self.sindy_coefficients = Xi.astype(np.float32)
        
        
        
        

class DatasetConstructor:
    def __init__(self, 
                input_dim=128,
                interp_dt=0.01,
                savgol_interp_coefs=[21, 3],
                interp_kind='cubic',
                future_steps=10):

        self.input_dim = input_dim
        self.interpolate = interpolate 
        self.interp_dt = interp_dt 
        self.savgol_interp_coefs = savgol_interp_coefs
        self.interp_kind = interp_kind
        self.future_steps = future_steps

    def get_data(self):
        return {
            't': self.t,
            'x': self.x,
            'dx': self.dx,
            'z': self.z,
            'dz': self.dz,
            'sindy_coefficients': self.sindy_coefficients
        }
    
    def build_solution(self, data):
        dt = data['dt']
        if 'time' in data.keys():
            times = data['time']
        elif 'dt' in data.keys():
            times = []
            for xr in data['x']:
                times.append(np.linspace(0, dt*len(xr), len(xr), endpoint=False))
        
        x = data['x']
        if 'dx' in data.keys():
            dx = data['dx']
        else:
            dx = [np.gradient(xr, dt) for xr in x]
        
                    
        n = self.input_dim 
        if self.future_steps > 0:
            n += self.future_steps
        n_delays = n
        xic = []
        dxic = []
        for j, xr in enumerate(x):
            print(j)
            print(len(xr))
            print(xr)
            n_steps = len(xr) - n
            print(f"n_steps: {n_steps}")
            xj = np.zeros((n_steps, n_delays))
            dxj = np.zeros((n_steps, n_delays))
            for k in range(n_steps):
                xj[k, :] = xr[k:n_delays+k]
                dxj[k, :] = dx[j][k:n_delays+k]
            xic.append(xj)
            dxic.append(dxj)
        H = np.vstack(xic)
        dH = np.vstack(dxic)
        
        self.t = np.hstack(times)
        self.x = H.T
        self.dx = dH.T
        self.z = np.hstack(x) 
        self.dz = np.hstack(dx)
        self.sindy_coefficients = None # unused
                
#         # Align times
#         for i in range(1, n_realizations):
#             if times[i] - times[i-1] >= dt*2:
#                 new_time[i] = new_time[i-1] + dt
        
class DatasetConstructorMulti:
    def __init__(self, 
                 input_dim=128,
                 interp_dt=0.01,
                 savgol_interp_coefs=[21, 3],
                 interp_kind='cubic',
                 future_steps=10):

        self.input_dim = input_dim
        self.interp_dt = interp_dt 
        self.savgol_interp_coefs = savgol_interp_coefs
        self.interp_kind = interp_kind
        self.future_steps = future_steps

    def get_data(self):
        return {
            't': self.t,
            'x': self.x,
            'dx': self.dx,
            'z': self.z,
            'dz': self.dz,
            'sindy_coefficients': self.sindy_coefficients
        }
    
    def build_solution(self, data):
        n_realizations = len(data['input_data'][0])  # Assuming 'x' is a list of lists
        n_variables = len(data['input_data'])  # Number of time series
        dt = data['dt']
        
        if 'time' in data.keys():
            times = data['time']
        elif 'dt' in data.keys():
            times = np.linspace(0, dt * len(data['input_data']), len(data['input_data']), endpoint=False)
        
        input_data = data['input_data']
        deriv_data = []
        for x_var in input_data:
            dx = [np.gradient(xr, dt) for xr in x_var]
            deriv_data.append(dx)
        
        n = self.input_dim 
        if self.future_steps > 0:
            n += self.future_steps
        n_delays = n
        xic = []
        dxic = []

        for i in range(n_variables):
            xic_var = []
            dxic_var = []
            for j, xr in enumerate(input_data[i]):
                n_steps = len(xr) - n_delays
                xj = np.zeros((n_steps, n_delays))
                dxj = np.zeros((n_steps, n_delays))
                for k in range(n_steps):
                    xj[k, :] = xr[k:n_delays+k]
                    dxj[k, :] = deriv_data[i][j][k:n_delays+k]
                xic_var.append(xj)
                dxic_var.append(dxj)
            
            # Stack all series for this variable
            xic.append(np.vstack(xic_var))
            dxic.append(np.vstack(dxic_var))

        # Stack all variables side by side
        H = np.stack(xic, axis=-1)  # Shape: (batch, n_delays, n_variables)
        dH = np.stack(dxic, axis=-1)

        # Transpose to get (batch, n_variables, n_delays)
        H = np.transpose(H, (0, 2, 1))
        dH = np.transpose(dH, (0, 2, 1))

        self.t = times
        self.x = H
        self.dx = dH
        self.z = np.stack([np.concatenate(x_var) for x_var in input_data], axis=-1)
        self.dz = np.stack([np.concatenate(dx_var) for dx_var in deriv_data], axis=-1)
        self.sindy_coefficients = None  # unused


class TemporalFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=100, min_periods=10, scaling_method='rolling', prediction_horizon=8):
        self.window_size = window_size
        self.min_periods = min_periods
        self.scaling_method = scaling_method
        self.prediction_horizon = prediction_horizon
        self.feature_scalers = {}

    def fit(self, X, y=None):
        # X should be a pandas DataFrame
        for column in X.columns:
            if self.scaling_method == 'expanding':
                self.feature_scalers[column] = {
                    'mean': X[column].expanding(min_periods=self.min_periods).mean(),
                    'std': X[column].expanding(min_periods=self.min_periods).std()
                }
            elif self.scaling_method == 'rolling':
                ## double check if this is correct
                self.feature_scalers[column] = {
                    'mean': X[column].rolling(window=self.window_size, min_periods=self.min_periods).mean(),
                    'std': X[column].rolling(window=self.window_size, min_periods=self.min_periods).std()
                }
        return self

    def transform(self, X):
        X_scaled = X.copy()
        for column in X.columns:
            if column in self.feature_scalers:
                mean = self.feature_scalers[column]['mean']
                std = self.feature_scalers[column]['std']
                
                # Update rolling statistics
                # if self.scaling_method == 'rolling':
                #     mean = mean.append(X[column].rolling(window=self.window_size, min_periods=self.min_periods).mean())
                #     std = std.append(X[column].rolling(window=self.window_size, min_periods=self.min_periods).std())
                #     self.feature_scalers[column]['mean'] = mean
                #     self.feature_scalers[column]['std'] = std
                
                # Apply normalization using the updated statistics
                X_scaled[column] = (X[column] - mean) / (std + 1e-8)
                
                # Handle NaNs
                X_scaled[column] = X_scaled[column].fillna(method='ffill').fillna(0)
        
        return X_scaled


    def inverse_transform(self, X, evaluation_column, alert_index, model_hyperparameters):
        X_inverse = X.cpu().detach().numpy()
        if evaluation_column == 'h':
            target = X_inverse[:, 1]
        elif evaluation_column == 'l':
            target = X_inverse[:, 2]

        # try:
        mean = self.feature_scalers[evaluation_column]['mean'].reset_index(drop=True)
        std = self.feature_scalers[evaluation_column]['std'].reset_index(drop=True)
        # instance_start = alert_index - model_hyperparameters['context_length']
        instance_std = std[alert_index:alert_index + model_hyperparameters['prediction_horizon']]
        instance_mean = mean[alert_index:alert_index + model_hyperparameters['prediction_horizon']]

        unscaled_target = (target * instance_std) + instance_mean
        # except:
        #     print(f"Error in inverse transform for column {column}")
        return unscaled_target
    

def scale_dataset(data, params):
    temporal_scaler = TemporalFeatureScaler(window_size=params['input_dim'], min_periods=10, scaling_method='rolling', prediction_horizon=params['future_steps'])
    scaled_data = temporal_scaler.fit_transform(data)
    return scaled_data
