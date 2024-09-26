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


KEY = "XpqF6xBLLrj6WALk4SS1UlkgphXmHQec"

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
    if symbol == 'META':
        date = datetime.strptime(from_str, "%Y-%m-%d")
        if date < datetime(2022,6,9):
            symbol = 'FB'
        else:
            symbol = 'META'
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_str}/{to_str}?adjusted=true&sort=asc&limit=50000&apiKey={KEY}"
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
    except Exception as e:
        print(f"call polygon {e}")
        print(f"symbol {symbol}, dates {from_str} -  {to_str}, timespan {timespan}, multiplier {multiplier}")

    return results_df


