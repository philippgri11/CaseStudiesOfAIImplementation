import os
from datetime import datetime
import joblib
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import yaml
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from wandb.integration.xgboost import WandbCallback
import wandb
from src.evaluation import evaluateXGBModel, evaluateLSTMModel, evaluateXGBModelWithCV
from src.loadData import getData
from src.preprocessing import preprocessingXGBoost, preprocessingLSTM
from wandb.keras import WandbCallback as WandbCallbackKeras
from src.sweep_config import getSweepIDLSTM, getSweepIDXGBoost
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)
    XGBRegressorModelParams = param['XGBRegressorModelParams']
    DNNModelParams = param['DNN_model_params']
    preprocessingParam = param['preprocessing']
    LSTMpreprocessing = param['LSTMpreprocessing']
    LSTM_model_params = param['LSTM_model_params']


def get_value(key, group):
    if wandb.config.get(key) is not None:
        return wandb.config.get(key)
    return param.get(group).get(key)


def trainXGBRegressor():
    with wandb.init(project='CaseStudiesOfAIImplementation', entity='philippgrill') as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        with open('../params.yaml', 'r') as file:
            param = yaml.safe_load(file)
        dataset = wandb.config.get('dataset') or param.get('dataset')
        preprocessingParam = {
            'test_size': get_value('test_size', 'preprocessing'),
            'val_size': get_value('val_size', 'preprocessing'),
            'colums': get_value('colums', 'preprocessing'),
            'shifts': get_value('shifts', 'preprocessing'),
            'negShifts': get_value('negShifts', 'preprocessing'),
            'enable_daytime_index': get_value('enable_daytime_index', 'preprocessing'),
            'monthlyCols': get_value('monthlyCols', 'preprocessing'),
            'keepMonthlyAvg': get_value('keepMonthlyAvg', 'preprocessing'),
            'dailyCols': get_value('dailyCols', 'preprocessing'),
            'keepDailyAvg': get_value('keepDailyAvg', 'preprocessing'),
            'loadLag': get_value('loadLag', 'preprocessing')
        }

        XGBRegressorModelParams = {
            'learning_rate': get_value('learning_rate', 'XGBRegressorModelParams'),
            'max_depth': get_value('max_depth', 'XGBRegressorModelParams'),
            'colsample_bytree': get_value('colsample_bytree', 'XGBRegressorModelParams'),
            'min_child_weight': get_value('min_child_weight', 'XGBRegressorModelParams'),
            'subsample': get_value('subsample', 'XGBRegressorModelParams'),
            'reg_alpha': get_value('reg_alpha', 'XGBRegressorModelParams'),
            'reg_lambda': get_value('reg_lambda', 'XGBRegressorModelParams'),
            'n_estimators': get_value('n_estimators', 'XGBRegressorModelParams'),
            'tree_method': get_value('tree_method', 'XGBRegressorModelParams'),
            'eval_metric': get_value('eval_metric', 'XGBRegressorModelParams'),
        }
        print(XGBRegressorModelParams)
        print(preprocessingParam)
        print(dataset)
        wandb.config.update(XGBRegressorModelParams)
        wandb.config.update({'dataset': dataset})
        wandb.config.update(preprocessingParam)
        data = getData(dataset)
        xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)
        # define model
        model = xgb.XGBRegressor(**XGBRegressorModelParams)

        model.fit(xTrain, yTrain, verbose=True, eval_set=[(xVal, yVal)], callbacks=[WandbCallback()]
                  )
        # save
        path = getPath()
        model.save_model(path)
        evaluateXGBModel(model=model, givenPreprocessingParam=preprocessingParam)
        evaluateXGBModelWithCV(model=model, givenPreprocessingParam=preprocessingParam)
        return path


def getPath():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'model_{timestamp}.json'
    model_dir = '../models'
    path = os.path.join(model_dir, filename)
    print(f'saved to {path}')
    return path


def trainXGBRegressorHP():
    data = getData(param['dataset'])
    print(preprocessingParam)

    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)

    param_grid = {
        'n_estimators': [1500],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01],
        'colsample_bytree': [0.3, 0.6, 0.9],
        'min_child_weight': [1, 4, 6],
        'subsample': [0.6, 0.8, 0.9],
        'reg_alpha': [0.1, 0.3, 0.5],
        'reg_lambda': [1.0, 1.5, 2],
        'eval_metric': ['mphe']
    }
    model = xgb.XGBRegressor(**XGBRegressorModelParams)

    grid_search = HalvingGridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),  # Benutzerdefinierter Scorer
        verbose=1
    )

    grid_search.fit(xTrain, yTrain)

    print('Best parameters found: ', grid_search.best_params_)
    cv_results = grid_search.cv_results_
    for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
        print(f'Mean CV Score: {mean_score:.2f}, Parameters: {params}')
    print(max(cv_results['mean_test_score']))
    best_model = grid_search.best_estimator_
    path = getPath()
    best_model.save_model(path)
    return path


def trainDNN():
    with wandb.init(project='CaseStudiesOfAIImplementation', entity='philippgrill') as run:
        data = getData(param['dataset'])
        xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(12,), name='input_layer'),
            tf.keras.layers.Dense(32, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(64, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(64, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(128, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(64, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(32, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(16, activation=DNNModelParams['activation']),
            tf.keras.layers.Dense(1, activation=DNNModelParams['activation'])
        ])
        model.compile(optimizer=DNNModelParams['optimizer'],
                      loss=DNNModelParams['loss'])

        model.fit(xTrain, yTrain, epochs=DNNModelParams['epochs'], batch_size=DNNModelParams['batch_size'],
                  validation_data=(xTest, yTest), callbacks=[WandbCallbackKeras()])
        evaluation = model.evaluate(xTest, yTest)
        print(f'Evaluation Loss: {evaluation}')
        path = getPath()
        model.save(path)
        return path


def trainLSTM():
    with wandb.init(project='CaseStudiesOfAIImplementation', entity='philippgrill') as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        dataset = wandb.config.get('dataset') or param.get('dataset')
        data = getData(dataset)
        LSTMpreprocessing = {
            'test_size': get_value('test_size', 'LSTMpreprocessing'),
            'shifts': get_value('shifts', 'LSTMpreprocessing'),
            'enable_daytime_index': get_value('enable_daytime_index', 'LSTMpreprocessing'),
            'monthlyCols': get_value('monthlyCols', 'LSTMpreprocessing'),
            'keepMonthlyAvg': get_value('keepMonthlyAvg', 'LSTMpreprocessing'),
            'dailyCols': get_value('dailyCols', 'LSTMpreprocessing'),
            'keepDailyAvg': get_value('keepDailyAvg', 'LSTMpreprocessing')
        }
        LSTMModelParams = {
            'epochs': get_value('epochs', 'LSTM_model_params'),
            'batch_size': get_value('batch_size', 'LSTM_model_params'),
            'learning_rate': get_value('learning_rate', 'LSTM_model_params'),
            'metrics': get_value('metrics', 'LSTM_model_params'),
            'dropout': get_value('dropout', 'LSTM_model_params')
        }
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            verbose=1,
            min_delta=0.01,
            restore_best_weights=True)

        print(LSTMpreprocessing)
        print(LSTMModelParams)
        print(dataset)
        wandb.config.update(LSTMpreprocessing)
        wandb.config.update({'dataset': dataset})
        wandb.config.update(LSTMModelParams)

        xTest, yTest, xTrain, yTrain = preprocessingLSTM(data, **LSTMpreprocessing)
        n_timesteps, n_features = xTrain.shape[1], xTrain.shape[2]
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
            tf.keras.layers.Dropout(LSTMModelParams['dropout']),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(LSTMModelParams['dropout']),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LSTMModelParams['learning_rate']),
                      loss=tf.keras.losses.Huber(),
                      metrics=LSTMModelParams['metrics'])

        model.fit(xTrain, yTrain, epochs=LSTMModelParams['epochs'], batch_size=LSTMModelParams['batch_size'],
                  validation_data=(xTest, yTest), callbacks=[WandbCallbackKeras(), earlyStopping])
        evaluation = model.evaluate(xTest, yTest)
        print(f'Evaluation Loss: {evaluation}')
        path = getPath()
        model.save(path)
        mse = evaluateLSTMModel(model=model, givenPreprocessingParam =  LSTMpreprocessing)
        wandb.log({"mse": mse})
        return path


def trainARDRegressor():
    print(param)
    data = getData(param['dataset'])
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)

    model = BayesianRidge(
        n_iter=1000
    )
    model.fit(xTrain, yTrain)
    path = getPath()
    joblib.dump(model, os.path.join(path))  # Speichern des Modells als ard_model.pkl im angegebenen Verzeichnis

    return path


def trainARDRegressorHP():
    print(param)
    data = getData(param['dataset'])
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)

    model = ARDRegression(
    )
    param_grid = {
        'n_iter': [100, 300, 500, 1000, 1500],
        'tol': [1e-3, 1e-4, 1e-5],
        'alpha_1': [1e-6, 1e-7, 1e-8],
        'alpha_2': [1e-6, 1e-7, 1e-8],
        'lambda_1': [1e-6, 1e-7, 1e-8],
        'lambda_2': [1e-6, 1e-7, 1e-8],
        'compute_score': [True, False]
    }
    grid_search = HalvingGridSearchCV(model,
                                      param_grid,
                                      cv=5,
                                      verbose=2,
                                      scoring=make_scorer(mean_squared_error,
                                                          greater_is_better=False),
                                      )
    grid_search.fit(xTrain, yTrain)
    print('Best parameters found: ', grid_search.best_params_)

    best_model = grid_search.best_estimator_

    path = getPath()
    joblib.dump(best_model, os.path.join(path))

    return path


def trainKNeighborsRegressor():
    print(param)
    data = getData(param['dataset'])
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)

    model = KNeighborsRegressor(
        n_neighbors=100,
        weights='distance',
        algorithm='kd_tree',
        p=1,
        leaf_size=60
    )
    model.fit(xTrain, yTrain)
    path = getPath()
    joblib.dump(model, os.path.join(path))

    return path


def trainKNeighborsRegressorHP():
    print(param)
    data = getData(param['dataset'])
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessingXGBoost(data, **preprocessingParam)

    model = KNeighborsRegressor()

    param_grid = {
        'n_neighbors': [20, 25, 50, 75, 100, 200],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
        'leaf_size': [15, 30, 60, 90]
    }

    grid_search = HalvingGridSearchCV(model,
                                      param_grid,
                                      cv=5,
                                      verbose=2,
                                      scoring=make_scorer(mean_squared_error,
                                                          greater_is_better=False),  # Benutzerdefinierter Scorer
                                      )

    grid_search.fit(xTrain, yTrain)
    print('Best parameters found: ', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    path = getPath()
    joblib.dump(best_model, os.path.join(path))  # Speichern des Modells als ard_model.pkl im angegebenen Verzeichnis

    return path


if __name__ == "__main__":
    wandb.agent(getSweepIDXGBoost(), trainXGBRegressor)
    # wandb.agent(getSweepIDXGBoost(), trainXGBRegressor)
    # trainLSTM()
    # trainXGBRegressor()
    # evaluateLSTMModel("..\models\model_2023-10-17_21-43-29.json")
