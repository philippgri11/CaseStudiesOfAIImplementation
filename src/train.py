import os
from datetime import datetime

import xgboost as xgb
import yaml
from sklearn.metrics import make_scorer
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import optimizers, models, layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from src.loadData import splitData, readLoadCurveOnetToDataframe, getData

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)
    XGBRegressorModelParams = param['XGBRegressor_model_params']
    DNNModelParams = param['DNN_model_params']

def trainXGBRegressor():
    xTest, yTest, xTrain, yTrain = getData(param['dataset'], param['test_size'])

    # define model
    model = xgb.XGBRegressor(**XGBRegressorModelParams)
    model.fit(xTrain, yTrain, verbose=True, eval_set=[(xTest, yTest)])

    # save
    path = getPath()
    # Speichern des Modells in einer Datei
    model.save_model(path)
    return path

def getPath():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'model_{timestamp}.json'
    model_dir = '..\models'
    path = os.path.join(model_dir, filename)
    print(f'saved to {path}')
    return path

def findBestParameter():
    xTest, yTest, xTrain, yTrain = getData(param['dataset'], param['test_size'])

    param_grid = {
        'n_estimators': [500],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01],
        'colsample_bytree': [0.3, 0.9],
        'min_child_weight': [1, 4],
        'subsample': [0.6, 0.9],
        'reg_alpha': [0.1, 0.5],
        'reg_lambda': [1.0, 1.5],
        'eval_metric': ['rmse', 'mphe']
    }
    model = xgb.XGBRegressor(**XGBRegressorModelParams)

    grid_search = HalvingGridSearchCV(
        model,
        param_grid,
        cv=2,
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
    xTest, yTest, xTrain, yTrain = getData(param['dataset'], param['test_size'])
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

    model.fit(xTrain, yTrain, epochs=DNNModelParams['epochs'], batch_size=DNNModelParams['batch_size'], validation_data=(xTest, yTest))
    evaluation = model.evaluate(xTest, yTest)
    print(f'Evaluation Loss: {evaluation}')
    path = getPath()
    model.save(path)
    return path
