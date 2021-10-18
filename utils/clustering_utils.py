from typing import Union
import numpy as np
import pandas as pd
import joblib

from glob import glob
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import pathlib
DATA_DIR = pathlib.Path.cwd()/'data/input'
OUT_DIR = pathlib.Path.cwd()/'data/output'

import sys 
sys.path.append(str(pathlib.Path.cwd()/'utils'))
from utils.misc_utils import load_train_test



def generate_return_corr():
    train, test = load_train_test()
    pivot_volatilty = (
        train
        .drop('id', axis=1)
        .pivot_table(index='time_id', columns = 'stock_id', values = 'target')
        .fillna(method='bfill')
        )
    
    pivot_volatilty_corr = pivot_volatilty.corr()
    return pivot_volatilty_corr

def best_cluster_kmeans(
    cluster_df: pd.DataFrame, 
    k: tuple = (2, 15),
    random_state: int = 123,
    **kwargs
    ):
    model = KMeans(random_state = random_state)
    elbow = KElbowVisualizer(
        model, 
        k=k, 
        **kwargs
        )

    elbow.fit(cluster_df)   
    optimal_n_clusters = elbow.elbow_value_
    optimal_k_means = KMeans(n_clusters=optimal_n_clusters, random_state = random_state)
    optimal_k_means.fit_transform(cluster_df)
    stock_clusters = optimal_k_means.predict(cluster_df)
    
    return stock_clusters, optimal_k_means


def train_and_dump_clusters(
    cluster_df: pd.DataFrame = None, 
    savepath: Union[pathlib.Path, str] = str(OUT_DIR),
    clustering_name: str = 'cluster_result',
    id_col: str = None,
    **kwargs
    ):
    
    if cluster_df is None: 
        cluster_df = generate_return_corr()
    
    stock_clusters, _ = (
        best_cluster_kmeans(
        cluster_df = cluster_df, 
        **kwargs
        )
    )
    
    if id_col is None:
        cluster_result_dict = dict(zip(cluster_df.index, stock_clusters))
    else:
        cluster_result_dict = dict(zip(cluster_df[id_col], stock_clusters))
    savepath = f'{str(savepath)}/clusters/{clustering_name}.pkl'
    joblib.dump(cluster_result_dict, savepath)
    