import numpy as np
import pandas as pd
import joblib

import os
from tqdm import tqdm
import pathlib
from joblib import Parallel, delayed

from utils.misc_utils import fullrange, realized_volatility, log_return, get_stock_path, load_parquet_file, load_parquet_files, load_train_test
from utils.clustering_utils import train_and_dump_clusters

competition = 'optiver-realized-volatility-prediction'
DATA_DIR = pathlib.Path.cwd()/'data/input'
OUT_DIR = pathlib.Path.cwd()/'data/output'

training_target, test = load_train_test()

##########################################################################################
##################################### Feature Dict #######################################
##########################################################################################

book_aggregation = {
    'wap1': [np.mean, np.std, fullrange], 
    'wap2': [np.mean, np.std, fullrange], 
    'wap_gap': [np.mean, np.std, fullrange], 
    'log_return_1': [fullrange, np.sum, np.mean, realized_volatility], 
    'log_return_2': [fullrange, np.sum, np.mean, realized_volatility], 
    'log_return_gap': [fullrange, np.sum, np.mean, realized_volatility], 
    'bid_ask_price_spread_1': [np.mean, np.std, fullrange, realized_volatility],
    'bid_ask_price_spread_2': [np.mean, np.std, fullrange, realized_volatility],
    'bid_ask_size_spread_1': [np.mean, np.std, fullrange, realized_volatility],
    'bid_ask_size_spread_2': [np.mean, np.std, fullrange, realized_volatility]
    }

trade_aggregation = {
    'volume': [np.mean, np.sum, np.std], 
    'price': [np.mean, np.std], 
    'order_count': [np.mean, np.sum, np.std],
    'trade_return': [fullrange, np.sum, np.mean, realized_volatility]
    }

time_agg_trade = {
    'volume_mean': [np.mean, np.sum, np.std], 
    'price_mean': [np.mean, np.std], 
    'order_count_mean': [np.mean, np.sum, np.std]
    }

time_agg_book = {
    'wap1_std': [np.mean], 
    'wap2_std': [np.mean], 
    'wap_gap_std': [np.mean], 
    'log_return_1_realized_volatility': [np.mean], 
    'log_return_2_realized_volatility': [np.mean], 
    'log_return_gap_realized_volatility': [np.mean], 
    'log_return_1_sum': [np.mean], 
    'log_return_2_sum': [np.mean] 
    }

stock_agg_trade = {
    'volume_mean': [np.mean,np.std], 
    'price_mean': [np.mean, np.std], 
    'order_count_mean': [np.mean, np.std]
    }

stock_agg_book = {
    'wap1_std': [np.std], 
    'wap2_std': [np.std], 
    'wap_gap_std': [np.std], 
    'log_return_1_realized_volatility': [np.std], 
    'log_return_2_realized_volatility': [np.std], 
    'log_return_gap_realized_volatility': [np.std], 
    'log_return_1_sum': [np.std], 
    'log_return_2_sum': [np.std] 
    }

##########################################################################################
##################################### Utils Function #####################################
##########################################################################################

def generate_features_book_data(df):
    df['wap1'] = (df['bid_price1'] * df['ask_size1'] +  df['ask_price1'] * df['bid_size1']) / (df['bid_size1']+ df['ask_size1'])
    df['wap2'] = (df['bid_price2'] * df['ask_size2'] +  df['ask_price2'] * df['bid_size2']) / (df['bid_size2']+ df['ask_size2'])
    df['wap_gap'] = np.abs(df['wap1'] -  df['wap2'])
    df['log_return_1'] = log_return(df['wap1']).fillna(0)
    df['log_return_2'] = log_return(df['wap2']).fillna(0)
    df['log_return_gap'] =  np.abs(df['log_return_1'] -  df['log_return_2'])
    df['bid_ask_price_spread_1'] = df['ask_price1'] - df['bid_price1']
    df['bid_ask_price_spread_2'] = df['ask_price2'] - df['bid_price2']
    df['bid_ask_size_spread_1'] = df['ask_size1'] - df['bid_size1']
    df['bid_ask_size_spread_2'] = df['ask_size2'] - df['bid_size2']
    return df

def generate_features_trade_data(df):
    df['volume'] = df['price'] * df['size']
    df['trade_return'] = log_return(df['price']).fillna(0)
    return df


def groupby_and_aggregate(df, agg_col, agg_dict, suffix=''):
    agg_df = df.groupby(agg_col).agg(agg_dict)
    agg_df.columns = [col[0] + '_' + col[1] + suffix for col in agg_df.columns]
    return agg_df


##########################################################################################
##################################### Full Feature Eng Pipeline ##########################
##########################################################################################

def full_feature_engineering(
    stock_ids: list = [],
    training_target: pd.DataFrame = training_target, 
    book_aggregation: dict = book_aggregation,
    trade_aggregation: dict = trade_aggregation,
    time_agg_book: dict = time_agg_book,
    time_agg_trade: dict = time_agg_trade,
    training: bool = True, 
    summarize_by_cluster: bool = True, 
    lower_seconds_cutoff: int = 0,
    upper_seconds_cutoff: int = 600
    ): 
    processed_stock_data_list = []
    

    def individual_stock_process(stock_id):
        if training:
            book = load_parquet_file(str(DATA_DIR)+ f'/book_train.parquet/stock_id={stock_id}')
            trade = load_parquet_file(str(DATA_DIR)+ f'/trade_train.parquet/stock_id={stock_id}')

        else: 
            book = load_parquet_file(str(DATA_DIR)+ f'/book_test.parquet/stock_id={stock_id}')     
            trade = load_parquet_file(str(DATA_DIR)+ f'/trade_test.parquet/stock_id={stock_id}')     

        # Important to generate the features first in order to have return for the first non-zero time cutoff
        book = generate_features_book_data(book)
        trade = generate_features_trade_data(trade)
                                             
        book = book.loc[(book.seconds_in_bucket>=lower_seconds_cutoff) & (book.seconds_in_bucket<upper_seconds_cutoff)]
        trade = trade.loc[(trade.seconds_in_bucket>=lower_seconds_cutoff) & (trade.seconds_in_bucket<upper_seconds_cutoff)]
        
        agg_book_data = groupby_and_aggregate(book, agg_col = 'id', agg_dict=book_aggregation)
        agg_trade_data =  groupby_and_aggregate(trade, agg_col = 'id', agg_dict=trade_aggregation)
        
        # Fillna with 0 make sense because in the data where we don't have trade data that means no trading had occured
        merged_df = agg_book_data.merge(agg_trade_data, left_index=True, right_index=True, how='left').fillna(0).reset_index()
        merged_df['stock_id'] = merged_df['id'].apply(lambda x: int(x.split('-')[0]))
        merged_df['time_id'] = merged_df['id'].apply(lambda x: int(x.split('-')[1]))
        return merged_df
    
    processed_stock_data_list = Parallel(n_jobs=8)(delayed(individual_stock_process)(stock_id) for stock_id  in tqdm(stock_ids))
    processed_stock_data = pd.concat(processed_stock_data_list)

    time_agg_trade_data = groupby_and_aggregate(processed_stock_data, agg_col = 'time_id', agg_dict=time_agg_trade, suffix='_period')
    time_agg_book_data = groupby_and_aggregate(processed_stock_data, agg_col = 'time_id', agg_dict=time_agg_book, suffix='_period')
    time_agg = time_agg_book_data.merge(time_agg_trade_data, left_index=True, right_index=True).reset_index()

    if summarize_by_cluster:
        clustered_df_list = []
        cluster_result_path = OUT_DIR/'clusters/cluster_result.pkl'

        if not cluster_result_path.is_file():
            train_and_dump_clusters()
            
        cluster_results = pd.read_pickle(cluster_result_path)
        processed_stock_data['clusters'] = processed_stock_data.stock_id.map(cluster_results)
        for cluster in set(cluster_results.values()):
            df_by_cluster = processed_stock_data.loc[processed_stock_data.clusters==cluster]
            #cluster_agg_trade_data = groupby_and_aggregate(df_by_cluster, agg_col = 'time_id', agg_dict=time_agg_trade, suffix=f'_period_cluster_{cluster}')
            #cluster_agg_book_data = groupby_and_aggregate(df_by_cluster, agg_col = 'time_id', agg_dict=time_agg_book, suffix=f'_period_cluster_{cluster}')
            
            cluster_agg_trade_data = groupby_and_aggregate(df_by_cluster, agg_col = 'time_id', agg_dict=time_agg_trade, suffix=f'_period_cluster')
            cluster_agg_book_data = groupby_and_aggregate(df_by_cluster, agg_col = 'time_id', agg_dict=time_agg_book, suffix=f'_period_cluster')
            cluster_agg = cluster_agg_book_data.merge(cluster_agg_trade_data, left_index=True, right_index=True)
            cluster_agg['clusters'] = cluster
            clustered_df_list.append(cluster_agg)
        
        clustered_df = pd.concat(clustered_df_list, axis=0).reset_index()
        #clustered_df = pd.concat(clustered_df_list, axis=1).reset_index()
        #time_agg = time_agg.merge(clustered_df, on = 'time_id', how='left')
    
    stock_agg_trade_data = groupby_and_aggregate(processed_stock_data, agg_col = 'stock_id', agg_dict=stock_agg_trade, suffix='_stock')
    stock_agg_book_data = groupby_and_aggregate(processed_stock_data, agg_col = 'stock_id', agg_dict=stock_agg_book, suffix='_stock')
    stock_agg = stock_agg_book_data.merge(stock_agg_trade_data, left_index=True, right_index=True).reset_index()

    processed_stock_data = processed_stock_data.merge(time_agg, on='time_id', how='left').merge(stock_agg, on='stock_id', how='left')
    
    if summarize_by_cluster:
        processed_stock_data['clusters'] = processed_stock_data.stock_id.map(cluster_results)
        processed_stock_data = processed_stock_data.merge(clustered_df, on = ['time_id', 'clusters'], how='left').drop('clusters', axis=1)

    if training:
        processed_stock_data =  training_target.merge(processed_stock_data, on=['id', 'stock_id', 'time_id'], how='left')

    return processed_stock_data



def full_feature_engineering_by_cutoff(
    cutoffs: list = [(0, 300), (300, 600)],
    training: bool = True, 
    **kwargs
    ):
    dataset_list = [] 

    for cutoff in cutoffs: 
        fe_data = full_feature_engineering(
            **kwargs,
            training = training,
            lower_seconds_cutoff=cutoff[0], 
            upper_seconds_cutoff=cutoff[1])
        if training:
            fe_data.set_index(['id', 'time_id', 'stock_id', 'target'], inplace=True)
        else:
            fe_data.set_index(['id', 'time_id', 'stock_id'], inplace=True)

        fe_data.columns = [col + f'_{cutoff[0]}_{cutoff[1]}' for col in fe_data]
        
        dataset_list.append(fe_data)

    final_data_set = pd.concat(dataset_list, axis=1)
    return final_data_set.reset_index()

