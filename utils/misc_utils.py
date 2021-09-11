import numpy as np
import pandas as pd
import joblib

import os
from tqdm import tqdm
import pathlib


competition = 'optiver-realized-volatility-prediction'
DATA_DIR = pathlib.Path.cwd()/'data/input'
OUT_DIR = pathlib.Path.cwd()/'data/output'





################################################################################
##########################Utils Functions#######################################
################################################################################

def load_train_test():
    train = pd.read_csv(DATA_DIR/'train.csv')
    test = pd.read_csv(DATA_DIR/'test.csv')

    train['id'] = train["stock_id"].astype(str) + '-' + train["time_id"].astype(str)
    test['id'] = test["stock_id"].astype(str) + '-' + test["time_id"].astype(str)
    
    return train, test

def get_stock_path(
    stock_ids: list = [], 
    train: bool = True, 
    file_type: str = 'book'
):
    split_type = 'test'
    if train:
        split_type = 'train'
    
    paths = [path for path in DATA_DIR.parent.glob('*/*/*/*') 
        if 
        (split_type in str(path)) &
        (file_type in str(path))
        ]
    
    if len(stock_ids)>0:
        paths = {int(str(path).split('stock_id=')[1].split('\\')[0]): path for path in paths if int(str(path).split('stock_id=')[1].split('\\')[0]) in stock_ids}
    else:
        paths = {int(str(path).split('stock_id=')[1].split('\\')[0]): path for path in paths}
    return paths


def load_parquet_file(path):  
    data = pd.read_parquet(path)
    data['stock_id'] = int(str(path).split('stock_id=')[1].split('\\')[0])
    data['id'] = data['stock_id'].astype(str) + '-' + data['time_id'].astype(str)
    return data 

def load_parquet_files(
    paths = None, 
    stock_ids: list = [], 
    train: bool = True, 
    file_type: str = 'book'
):  
    df_list = [] 
    
    if paths is None: 
        paths = get_stock_path(stock_ids, train, file_type).values()

    for path in tqdm(paths):
        data = load_parquet_file(path)
        df_list.append(data)

    return pd.concat(df_list)



############################### Calculations ################################
def log_return(list_stock_prices):
    log_return = np.log(list_stock_prices).diff() 
    return log_return
    
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def fullrange(series):
    return series.max() - series.min()

