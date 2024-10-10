import itertools
import numpy as np
import random 
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.utils.data import TensorDataset, DataLoader

KEY = "XpqF6xBLLrj6WALk4SS1UlkgphXmHQec"

class TemporalFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=100, min_periods=10, scaling_method='rolling'):
        self.window_size = window_size
        self.min_periods = min_periods
        self.scaling_method = scaling_method
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

def get_hyperparameter_list(hyperparams):
    def dict_product(dicts):
        return [dict(zip(dicts, x)) for x in itertools.product(*dicts.values())]
    hyperparams_list = dict_product(hyperparams)
    random.shuffle(hyperparams_list)
    return hyperparams_list

def get_hankel(x, dimension, delays, skip_rows=1):
    if skip_rows>1:
        delays = len(x) - delays * skip_rows
    H = np.zeros((dimension, delays))
    for j in range(delays):
        H[:, j] = x[j*skip_rows:j*skip_rows+dimension]
    return H

def get_hankel_svd(H, reduced_dim):
    U, s, VT = np.linalg.svd(H, full_matrices=False)
    rec_v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
    return U, s, VT, rec_v


class CustomRetry(Retry):
    def is_retry(self, method, status_code, has_retry_after=False):
        """ Return True if we should retry the request, otherwise False. """
        if status_code != 200:
            return True
        return super().is_retry(method, status_code, has_retry_after)
    
def setup_session_retries(
    retries: int = 3,
    backoff_factor: float = 0.05,
    status_forcelist: tuple = (500, 502, 504),
):
    """
    Sets up a requests Session with retries.
    
    Parameters:
    - retries: Number of retries before giving up. Default is 3.
    - backoff_factor: A factor to use for exponential backoff. Default is 0.3.
    - status_forcelist: A tuple of HTTP status codes that should trigger a retry. Default is (500, 502, 504).

    Returns:
    - A requests Session object with retry configuration.
    """
    retry_strategy = CustomRetry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def execute_polygon_call(url):
    session = setup_session_retries()
    response = session.request("GET", url, headers={}, data={})
    return response 

def convert_timestamp_est(timestamp):
    # Create a naive datetime object from the UNIX timestamp
    dt_naive = datetime.utcfromtimestamp(timestamp)
    # Convert the naive datetime object to a timezone-aware one (UTC)
    dt_utc = pytz.utc.localize(dt_naive)
    # Convert the UTC datetime to EST
    dt_est = dt_utc.astimezone(pytz.timezone('US/Eastern'))
    
    return dt_est

def call_polygon(symbol,from_str,to_str,timespan,multiplier):
    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_str}/{to_str}?adjusted=true&sort=asc&limit=50000&apiKey={KEY}"
    next_url = url

    while next_url:
        # try:
        response = execute_polygon_call(url)
        response_data = json.loads(response.text)
        results = response_data['results']
        results_df = pd.DataFrame(results)
        results_df['t'] = results_df['t'].apply(lambda x: int(x/1000))
        results_df['date'] = results_df['t'].apply(lambda x: convert_timestamp_est(x))
        results_df['hour'] = results_df['date'].apply(lambda x: x.hour)
        results_df['day'] = results_df['date'].apply(lambda x: x.day)
        results_df['minute'] = results_df['date'].apply(lambda x: x.minute)
        results_df = results_df.loc[(results_df['hour'] >= 9) & (results_df['hour'] < 16)]
        results_df['range_vol'] = (results_df['h'] - results_df['l'])/results_df['o']
        all_results.append(results_df)
        # except Exception as e:
        #     print(f"call polygon {e}")
        #     print(f"symbol {symbol}, dates {from_str} -  {to_str}, timespan {timespan}, multiplier {multiplier}")

        if 'next_url' in response_data.keys():
            next_url = response_data['next_url']
            url = f"{next_url}&apiKey={KEY}"
        else:
            next_url = None

    full_results = pd.concat(all_results)
    return full_results

def get_hankel_svd(H, reduced_dim):
    U, s, VT = np.linalg.svd(H, full_matrices=False)
    rec_v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
    return U, s, VT, rec_v

def create_state_space_embedding(data, embedding_dimension, embedding_gap=1, row_offset=1, prediction_horizon=1):
    """
    Create state space embeddings from a time series dataset.
    
    Parameters:
    data (array-like): The input time series data
    num_embeddings (int): Number of delay embeddings (dimension of the state space)
    embedding_gap (int): Gap between successive elements in each embedding vector
    row_offset (int): Offset between rows in the resulting dataset
    
    Returns:
    np.array: Array of state space embedding vectors
    """
    data = np.array(data)
    n = len(data)
    
    # Calculate the size of each complete embedding vector
    vector_size = ((embedding_dimension + prediction_horizon) - 1) * embedding_gap + 1
    
    # Calculate the number of complete embedding vectors we can create
    num_vectors = (n - vector_size) // row_offset + 1
    
    # Initialize the output array
    embeddings = np.zeros((num_vectors, embedding_dimension))
    targets = np.zeros((num_vectors, prediction_horizon))
    
    for i in range(num_vectors):
        start = i * row_offset
        for j in range(embedding_dimension):
            embeddings[i, j] = data[start + j * embedding_gap]
        for k in range(prediction_horizon):
            targets[i, k] = data[(start+embedding_dimension) + k]
    
    return embeddings, targets



def build_dataset(model_hyperparameters, local_dataset, train_ratio=0.8):
    embedded_matrices = []
    target_matrices = []
    if local_dataset:
        scaled_data = pd.read_csv(local_dataset)
    else:
        raw_data = call_polygon('SPY','2010-01-01','2024-06-01','minute',15)
        scaled_data = scale_data(raw_data, model_hyperparameters['numerical_features'], model_hyperparameters['state_space_embedding'])
        scaled_data.to_csv('scaled_data_SPY.csv')

    model_hyperparameters['numerical_features'] = ['c','v','range_vol']
    # Create embeddings for each feature
    for feature in model_hyperparameters['numerical_features']:
        state_space_embeddings_matrix, target_matrix = create_state_space_embedding(
            scaled_data[feature].values, embedding_dimension=model_hyperparameters['state_space_embedding'], 
            embedding_gap=1, row_offset=4, prediction_horizon=model_hyperparameters['prediction_horizon']
            )
        
        embedded_matrices.append(state_space_embeddings_matrix)
        target_matrices.append(target_matrix)

    
    # Find the minimum number of examples across all features
    min_examples = min(emb.shape[0] for emb in embedded_matrices)
    # Combine embeddings into a single tensor
    combined_embeddings = np.stack([emb[:min_examples] for emb in embedded_matrices], axis=1)
    # Combine targets into a single tensor
    combined_targets = np.stack([tar[:min_examples] for tar in target_matrices], axis=1)

    print(len(combined_embeddings), len(combined_targets))
    print(combined_embeddings[:int(len(combined_embeddings)*train_ratio),:,:].shape)
    train_data = TensorDataset(
        torch.FloatTensor(combined_embeddings[:int(len(combined_embeddings)*train_ratio),:,:]), 
        torch.FloatTensor(combined_targets[:int(len(combined_targets)*train_ratio),:,:])
        )
    val_data = TensorDataset(
        torch.FloatTensor(combined_embeddings[int(len(combined_embeddings)*train_ratio):,:,:]), 
        torch.FloatTensor(combined_targets[int(len(combined_targets)*train_ratio):,:,:])
        )
    
    train_loader = DataLoader(train_data, batch_size=model_hyperparameters['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=model_hyperparameters['batch_size'])
    
    return train_loader, val_loader


def scale_data(data,features, embedding_dimension):
    min_periods = 10
    scaler = TemporalFeatureScaler(window_size=(embedding_dimension*2), min_periods=min_periods, scaling_method='rolling') 
    scaled_data = scaler.fit_transform(data[features])
    scaled_data = scaled_data.iloc[min_periods:]
    return scaled_data

if __name__ == "__main__":
    feature_data, target_data  = build_dataset(feature_columns=['c','v'], local_dataset=None, embedding_dimension=128, prediction_horizon=4)
    # print(raw_data.head())