import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from utils.misc_utils import rmspe_eval, rmspe_obj, rmspe
from utils.feature_engineering_utils import full_feature_engineering_by_cutoff, load_train_test
from utils.logging_utils import create_logger

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
logger = create_logger(export_log=False)

def optiver_train_and_log_experiment(run_name, cutoffs):
    #mlflow.set_tracking_uri("https://localhost:1111")
    logger.info(f'...Currently Running {run_name}...')

    mlflow_experiment_id = 0
    mlflow.start_run(run_name = run_name, experiment_id=mlflow_experiment_id)

    train, test = load_train_test()
    train_id = train.stock_id.unique()
    test_id = test.stock_id.unique()
    
    logger.info('...Currently Feature Engineering...')
    final_training_data = full_feature_engineering_by_cutoff(
        cutoffs = cutoffs,
        stock_ids = train_id,
        training=True)

    mlflow.log_param('Cutoffs', cutoffs)
    mlflow.log_param('Train Dataset Row/Columns', final_training_data.shape)
    mlflow.log_param('Test Dataset Row/Columns', final_training_data.shape)


    ########################Initial Model Training########################
    logger.info('...Currently Training First LGBM...')
    model_col = [col for col in final_training_data.columns if ('id' not in col) & ('target' not in col)]

    X_train, X_test, y_train, y_test = train_test_split(
                                            final_training_data.drop('target', axis=1)[model_col],
                                            final_training_data['target'],
                                            test_size=0.1
                                            )

    X_train, X_valid, y_train, y_valid = train_test_split(
                                            X_train,
                                            y_train,
                                            test_size=0.1
                                            )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    test_data = lgb.Dataset(X_test, label=y_test)

    parameters = {'verbosity': 0,
                    'n_jobs': -1,
                    'seed': 123}

    model = lgb.train(parameters,
                        train_data,
                        valid_sets=valid_data,
                        fobj = rmspe_obj,
                        feval = rmspe_eval,
                        num_boost_round=50000,
                        early_stopping_rounds=200)

    mlflow.log_metric('Test Score Full Model', rmspe(y_test, model.predict(X_test)))

    ########################LOFO Feature Selection#######################
    logger.info('...Currently Performing LOFO...')

    # extract a sample of the data
    sample_df = X_test.copy() 
    sample_df['target'] = y_test
    sample_df = sample_df.sample(frac=0.01, random_state=0)

    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    dataset = Dataset(df=sample_df, target="target", features=[col for col in sample_df.columns if col != 'target'])
    lofo_imp = LOFOImportance(dataset, cv=cv, scoring=rmspe_scorer)
    importance_df = lofo_imp.get_importance()
    selected_lofo_features = importance_df.loc[importance_df.importance_mean>0.001]['feature'].to_list()

    mlflow.log_param('Feature Importance DF', importance_df)
    mlflow.log_param('LOFO Selected Features', selected_lofo_features)

    ########################Final Model Training#######################
    logger.info('...Currently Training Final Model...\n')
    selected_lofo_features = importance_df.loc[importance_df.importance_mean>0.001]['feature'].to_list()
    X_train_lofo, X_test_lofo, y_train_lofo, y_test_lofo = train_test_split(
                                            final_training_data.drop('target', axis=1)[selected_lofo_features],
                                            final_training_data['target'],
                                            test_size=0.1
                                            )

    X_train_lofo, X_valid_lofo, y_train_lofo, y_valid_lofo = train_test_split(
                                            X_train_lofo,
                                            y_train_lofo,
                                            test_size=0.1
                                            )

    train_data_lofo = lgb.Dataset(X_train_lofo, label=y_train_lofo)
    valid_data_lofo = lgb.Dataset(X_valid_lofo, label=y_valid_lofo)
    test_data_lofo = lgb.Dataset(X_test_lofo, label=y_test_lofo)

    parameters = {'verbosity': 0,
                    'n_jobs': -1,
                    'seed': 123}

    model = lgb.train(parameters,
                        train_data_lofo,
                        valid_sets=valid_data_lofo,
                        fobj = rmspe_obj,
                        feval = rmspe_eval,
                        num_boost_round=50000,
                        early_stopping_rounds=200)


    mlflow.log_metric('Test Score Feature-Selected Model', rmspe(y_test_lofo, model.predict(X_test_lofo)))
    mlflow.end_run()