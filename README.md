# kaggle-optiver-volatility
Repository for codes developed for Kaggle's [Optiver Volatility Prediction Competition](https://www.kaggle.com/c/optiver-realized-volatility-prediction/overview)
The objective is to develop a model to predict the level of volatility in stock in the next 10 minuts given a specific 10 minutes time window

### Repo Structure and Files
* `train.py`: provide the full training pipeline 
* `notebook runner.ipynb`: contains run instances of the training pipeline
* `utils`: folder contains all the helper functions for the training pipeline
* `01_data_preprocessing` and `02_model_training`: contains breakdown of training pipeline for debugging and development purpose
* `requirements.txt`: contains packages necessary to reproduce the result

### Training Pipeline
The training pipeline takes in `time_cutoffs` argument to bin and aggregate each 600 seconds input time period into input features. The training pipeline are constructed as followed:
1. **Data Preprocessing**: Input Data is aggregated based on the `time_cutoffs` argument provided and a full set of feature engineering is performed.
2. **Feature Engineering**:  The first `LightGBM` model is trained with default hyperparameters and feature engineering perform with `LOFOImportance` to select only features with positive contributions
3. **Hyperparameter Tuning**:  Using the features selected,  `LightGBM` hyperparameters are tuned using `optuna` module.  The best hyperparameter is used in the final model. 
4. **Logging and Experiment Tracking**:  `MLFlow` is used to track and save each experiment for comparison purpose

