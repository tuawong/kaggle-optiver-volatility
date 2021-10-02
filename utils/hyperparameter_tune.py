from typing import Union

import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils.misc_utils import rmspe_eval, rmspe_obj, rmspe
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

fixed_params = {
    'verbosity': -1,
    'n_jobs': -1,
    'seed': 123, 
    'metric': 'rmse'
    }

def lightgbm_optuna_objective(
    trial, 
    X: pd.DataFrame, 
    y: Union[pd.Series, np.array], 
    fixed_params=fixed_params, 
    param_grid: dict = None, 
    random_state: int = 123
    ):
    
    cv = KFold(n_splits=5, random_state=random_state, shuffle=True)

    if param_grid is None:
        param_grid = {
            "num_iterations": trial.suggest_int("num_iterations", 14000, 50000, step=2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 8, 4088, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 500, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 10),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95),
            **fixed_params
        }
    else:
        param_grid = {
            **param_grid, 
            **fixed_params
        }
    

    pruning = LightGBMPruningCallback(trial, "rmse", valid_name='valid_1')
    cv_score_rmspe = [] 

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_valid = y[train_idx], y[test_idx]

        train_data_cv = lgb.Dataset(X_train, label=y_train)
        valid_data_cv = lgb.Dataset(X_valid, label=y_valid)
        
        model =  lgb.train(param_grid,
            train_set=train_data_cv,
            valid_sets=[train_data_cv, valid_data_cv],
            early_stopping_rounds=100,
            verbose_eval=0,   
            fobj = rmspe_obj,
            feval = rmspe_eval,
            callbacks=[pruning]
        )
        predictions = model.predict(X_valid)
        cv_score_rmspe.append(rmspe(predictions, y_valid))

    return np.mean(cv_score_rmspe)