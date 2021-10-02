import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import optuna
from optuna.integration import LightGBMPruningCallback
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

from sklearn.model_selection import KFold

from utils.misc_utils import rmspe_eval, rmspe_obj, rmspe
from utils.feature_engineering_utils import full_feature_engineering_by_cutoff, load_train_test
from utils.logging_utils import create_logger
from utils.hyperparameter_tune import lightgbm_optuna_objective

import mlflow

import pathlib
DATA_DIR = pathlib.Path.cwd()/'data/input'
OUT_DIR = pathlib.Path.cwd()/'data/output'

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
logger = create_logger(export_log=False)
seed = 123

def optiver_train_and_log_experiment(
    run_name: str,
    time_cutoffs: list,
    from_scratch: bool = False,
    lofo_threshold: float = 0, 
    hyp_tuning: bool = False,
    hyp_tuning_time: int = 3600, 
    input_hyp_params: dict = {}
    ) -> None:
    
    #mlflow.set_tracking_uri("https://localhost:1111")
    logger.info(f'...Currently Running {run_name}...')

    mlflow.set_experiment('kaggle_optiver_experiment')
    mlflow.start_run(run_name = run_name)
    
    train, test = load_train_test()
    train_id = train.stock_id.unique()
    test_id = test.stock_id.unique()
    
    logger.info('...Currently Feature Engineering...')
    file_path = OUT_DIR/f'final_preprocessed_data_{time_cutoffs}.pkl'
    if from_scratch:
        final_preprocessed_data = full_feature_engineering_by_cutoff(
            cutoffs = time_cutoffs,
            stock_ids = train_id,
            training=True)
        final_preprocessed_data.to_pickle(file_path)
    else: 
        final_preprocessed_data = pd.read_pickle(file_path)
    
    mlflow.log_param('Cutoffs', time_cutoffs)
    
    ########################Initial Model Training########################
    logger.info('...Currently Training First LGBM...')
    model_col = [col for col in final_preprocessed_data.columns if ('id' not in col) & ('target' not in col)]

    X_train, X_test, y_train, y_test = train_test_split(
                                            final_preprocessed_data.drop('target', axis=1)[model_col],
                                            final_preprocessed_data['target'],
                                            test_size=0.1, 
                                            random_state = seed
                                            )

    X_train, X_valid, y_train, y_valid = train_test_split(
                                            X_train,
                                            y_train,
                                            test_size=0.1,
                                            random_state = seed
                                            )
    
    mlflow.log_param('Columns', X_train.columns)
    mlflow.log_param('Train Dataset Row/Columns', X_train.shape)
    mlflow.log_param('Test Dataset Row/Columns', X_valid.shape)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    test_data = lgb.Dataset(X_test, label=y_test)

    parameters = {'verbosity': -1,
                    'n_jobs': -1,
                    'seed': 123}

    model = lgb.train(parameters,
                        train_data,
                        valid_sets=valid_data,
                        fobj = rmspe_obj,
                        feval = rmspe_eval,
                        verbose_eval=False,
                        num_boost_round=50000,
                        early_stopping_rounds=200)

    mlflow.log_metric('Train Score Full Model', rmspe(y_train, model.predict(X_train)))
    mlflow.log_metric('Test Score Full Model', rmspe(y_test, model.predict(X_test)))

    ########################LOFO Feature Selection#######################
    logger.info('...Currently Performing LOFO...')

    # extract a sample of the data
    sample_df = X_test.copy() 
    sample_df['target'] = y_test
    sample_df = sample_df.sample(frac=0.01, random_state=seed)

    cv = KFold(n_splits=4, shuffle=True, random_state=seed)
    dataset = Dataset(df=sample_df, target="target", features=[col for col in sample_df.columns if col != 'target'])
    lofo_imp = LOFOImportance(dataset, cv=cv, scoring=rmspe_scorer)
    importance_df = lofo_imp.get_importance()
    selected_lofo_features = importance_df.loc[importance_df.importance_mean>lofo_threshold]['feature'].to_list()
    artifact_path = mlflow.get_artifact_uri().split('///')[1]
    
    plot_importance(importance_df[:50], figsize=(12, 20))
    plt.tight_layout()  
    plt.savefig(artifact_path + '/importance_plot.png')
    plt.close()

    mlflow.log_param('LOFO Importance Selection Threshold', lofo_threshold)
    mlflow.log_param('LOFO Selected Features', selected_lofo_features)

    ########################Final Model Training#######################
    X_train_lofo, X_valid_lofo, X_test_lofo = X_train[selected_lofo_features], X_valid[selected_lofo_features], X_test[selected_lofo_features]

    train_data_lofo = lgb.Dataset(X_train_lofo, label=y_train)
    valid_data_lofo = lgb.Dataset(X_valid_lofo, label=y_valid)
    test_data_lofo = lgb.Dataset(X_test_lofo, label=y_test)

    parameters = {
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 123
        }
    
    if hyp_tuning:
        logger.info('...Currently Tuning Hyperparameter...')    
        
        optuna_obj = functools.partial(
            lightgbm_optuna_objective,
            X = final_preprocessed_data[selected_lofo_features], 
            y = final_preprocessed_data['target']
            )
        
        # Optuna tuning for hyperparameter
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=25))
        study.optimize(
            optuna_obj,
            timeout=hyp_tuning_time
            )

        parameters ={
            **study.best_params, 
            **parameters
            }
    
        logger.info('...Currently Training Final Model...')    
        model_final =  lgb.train(
            params = parameters,
            train_set= train_data_lofo,
            valid_sets= [train_data_lofo, valid_data_lofo],
            fobj = rmspe_obj,
            feval = rmspe_eval,
            verbose_eval = 0,
            early_stopping_rounds=200
             )
        mlflow.log_param('Tuning Time', hyp_tuning_time)

    else:
        if bool(input_hyp_params): 
            parameters ={
            **input_hyp_params, 
            **parameters
            }
        
        logger.info('...Currently Training Final Model...')    
        model_final = lgb.train(parameters,
                            train_set = train_data_lofo,
                            valid_sets=valid_data_lofo,
                            fobj = rmspe_obj,
                            feval = rmspe_eval,
                            verbose_eval = 0,
                            num_boost_round=50000,
                            early_stopping_rounds=200
                            )


    mlflow.log_param('LGMB Hyperparameter', parameters)
    mlflow.log_metric('Train Score Feature-Selected Model', rmspe(y_train, model_final.predict(X_train_lofo)))
    mlflow.log_metric('Test Score Feature-Selected Model', rmspe(y_test, model_final.predict(X_test_lofo)))
    mlflow.lightgbm.log_model(model_final, 'model')
    mlflow.end_run()
    logger.info('...Training Ends...\n')