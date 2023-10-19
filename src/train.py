import os
from datetime import datetime
import joblib
import tensorflow as tf
import xgboost as xgb
import yaml
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from wandb.integration.xgboost import WandbCallback
import wandb
from src.evaluation import evaluateXGBModel
from src.loadData import getData
from src.preprocessing import preprocessing
from src.sweep_config import sweep_id

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)
    XGBRegressorModelParams = param['XGBRegressorModelParams']
    DNNModelParams = param['DNN_model_params']
    preprocessingParam = param['preprocessing']


def get_value(key, group):
    return wandb.config.get(key) or param.get(group).get(key)


def trainXGBRegressor():
    with wandb.init(project='CaseStudiesOfAIImplementation', entity='philippgrill') as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        with open('../params.yaml', 'r') as file:
            param = yaml.safe_load(file)
        dataset = wandb.config.get('dataset') or param.get('dataset')
        preprocessingParam = {
            'test_size': get_value('test_size', 'preprocessing'),
            'colums': get_value('colums', 'preprocessing'),
            'shifts': get_value('shifts', 'preprocessing'),
            'negShifts': get_value('negShifts', 'preprocessing')
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
            'eval_metric': get_value('eval_metric', 'XGBRegressorModelParams')
        }
        print(XGBRegressorModelParams)
        print(preprocessingParam)
        print(dataset)
        wandb.config.update(XGBRegressorModelParams)
        wandb.config.update({'dataset': dataset})
        wandb.config.update(preprocessingParam)
        data = getData(dataset)
        xTest, yTest, xTrain, yTrain = preprocessing(data, **preprocessingParam)
        # define model
        model = xgb.XGBRegressor(**XGBRegressorModelParams)
        model.fit(xTrain, yTrain, verbose=True, eval_set=[(xTest, yTest)], callbacks=[WandbCallback(log_model=True)])
        # save
        path = getPath()
        model.save_model(path)
        mse = evaluateXGBModel(model=model, givenPreprocessingParam=preprocessingParam)
        wandb.log({"mse": mse})
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
    wandb.agent(sweep_id, trainXGBRegressor)
    trainXGBRegressor()
