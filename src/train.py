import os
from datetime import datetime

import joblib
import xgboost as xgb
import yaml
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.neighbors import KNeighborsRegressor

from src.evaluation import evaluateXGBModel, evaluateDNNModel, evaluateARDModel
from src.loadData import readLoadCurveOnetToDataframe, getData
from src.preprocessing import preprocessing

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)
    XGBRegressorModelParams = param['XGBRegressor_model_params']
    DNNModelParams = param['DNN_model_params']
    preprocessingParam = param['preprocessing']


def trainXGBRegressor():
    print(param)
    data = getData(param['dataset'])
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)
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


def trainXGBRegressorHP():
    data = getData(param['dataset'])
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)

    param_grid = {
        'n_estimators': [ 1500],
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
    data = getData(param['dataset'])
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)
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
              validation_data=(xTest, yTest))
    evaluation = model.evaluate(xTest, yTest)
    print(f'Evaluation Loss: {evaluation}')
    path = getPath()
    model.save(path)
    return path


def trainARDRegressor():
    print(param)
    data = getData(param['dataset'])
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)

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
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)

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
                                                          greater_is_better=False),  # Benutzerdefinierter Scorer
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
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)

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
    xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)

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
    path = trainXGBRegressorHP()
    evaluateXGBModel(path)