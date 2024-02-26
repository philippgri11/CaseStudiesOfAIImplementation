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
from wandb.keras import WandbCallback as WandbCallbackKeras

import wandb
from src.evaluation import evaluate_xgb_model, evaluate_lstm_model, evaluate_xgb_model_with_cv
from src.load_data import get_data
from src.preprocessing import preprocessing_xg_boost, preprocessing_lstm
from src.sweep_config import getSweepIDXGBoost

with open("../params.yaml", "r") as file:
    param = yaml.safe_load(file)
    xgb_regressor_model_params = param["XGBRegressorModelParams"]
    dnn_model_params = param["DNN_model_params"]
    preprocessing_param = param["preprocessing"]
    lstm_preprocessing = param["LSTMpreprocessing"]
    lstm_model_params = param["LSTM_model_params"]


def get_value(key, group):
    if wandb.config.get(key) is not None:
        return wandb.config.get(key)
    return param.get(group).get(key)


def trainXGBRegressor():
    with wandb.init(
        project="CaseStudiesOfAIImplementation", entity="philippgrill"
    ) as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        with open("../params.yaml", "r") as file:
            param = yaml.safe_load(file)
        dataset = wandb.config.get("dataset") or param.get("dataset")
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "colums": get_value("colums", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "negShifts": get_value("negShifts", "preprocessing"),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "monthlyCols": get_value("monthlyCols", "preprocessing"),
            "keepMonthlyAvg": get_value("keepMonthlyAvg", "preprocessing"),
            "dailyCols": get_value("dailyCols", "preprocessing"),
            "keepDailyAvg": get_value("keepDailyAvg", "preprocessing"),
            "loadLag": get_value("loadLag", "preprocessing"),
        }

        xgb_regressor_model_params = {
            "learning_rate": get_value("learning_rate", "XGBRegressorModelParams"),
            "max_depth": get_value("max_depth", "XGBRegressorModelParams"),
            "colsample_bytree": get_value(
                "colsample_bytree", "XGBRegressorModelParams"
            ),
            "min_child_weight": get_value(
                "min_child_weight", "XGBRegressorModelParams"
            ),
            "subsample": get_value("subsample", "XGBRegressorModelParams"),
            "reg_alpha": get_value("reg_alpha", "XGBRegressorModelParams"),
            "reg_lambda": get_value("reg_lambda", "XGBRegressorModelParams"),
            "n_estimators": get_value("n_estimators", "XGBRegressorModelParams"),
            "tree_method": get_value("tree_method", "XGBRegressorModelParams"),
            "eval_metric": get_value("eval_metric", "XGBRegressorModelParams"),
        }
        print(xgb_regressor_model_params)
        print(preprocessing_param)
        print(dataset)
        wandb.config.update(xgb_regressor_model_params)
        wandb.config.update({"dataset": dataset})
        wandb.config.update(preprocessing_param)
        data = get_data(dataset)
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
            data, **preprocessing_param
        )
        # define model
        model = xgb.XGBRegressor(**xgb_regressor_model_params)

        model.fit(
            x_train,
            y_train,
            verbose=True,
            eval_set=[(x_val, y_val)],
            callbacks=[WandbCallback()],
        )
        # save
        path = get_path()
        model.save_model(path)
        evaluate_xgb_model(model=model, given_preprocessing_param=preprocessing_param)
        evaluate_xgb_model_with_cv(model=model, given_preprocessing_param=preprocessing_param)
        return path


def get_path():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"model_{timestamp}.json"
    model_dir = "../models"
    path = os.path.join(model_dir, filename)
    print(f"saved to {path}")
    return path


def train_dnn():
    with wandb.init(
        project="CaseStudiesOfAIImplementation", entity="philippgrill"
    ) as run:
        data = get_data(param["dataset"])
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
            data, **preprocessing_param
        )
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(12,), name="input_layer"),
                tf.keras.layers.Dense(32, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(64, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(64, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(128, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(64, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(32, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(16, activation=dnn_model_params["activation"]),
                tf.keras.layers.Dense(1, activation=dnn_model_params["activation"]),
            ]
        )
        model.compile(
            optimizer=dnn_model_params["optimizer"], loss=dnn_model_params["loss"]
        )

        model.fit(
            x_train,
            y_train,
            epochs=dnn_model_params["epochs"],
            batch_size=dnn_model_params["batch_size"],
            validation_data=(x_test, y_test),
            callbacks=[WandbCallbackKeras()],
        )
        evaluation = model.evaluate(x_test, y_test)
        print(f"Evaluation Loss: {evaluation}")
        path = get_path()
        model.save(path)
        return path


def train_lstm():
    with wandb.init(
        project="CaseStudiesOfAIImplementation", entity="philippgrill"
    ) as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        dataset = wandb.config.get("dataset") or param.get("dataset")
        data = get_data(dataset)
        lstm_preprocessing = {
            "test_size": get_value("test_size", "LSTMpreprocessing"),
            "shifts": get_value("shifts", "LSTMpreprocessing"),
            "enable_daytime_index": get_value(
                "enable_daytime_index", "LSTMpreprocessing"
            ),
            "monthlyCols": get_value("monthlyCols", "LSTMpreprocessing"),
            "keepMonthlyAvg": get_value("keepMonthlyAvg", "LSTMpreprocessing"),
            "dailyCols": get_value("dailyCols", "LSTMpreprocessing"),
            "keepDailyAvg": get_value("keepDailyAvg", "LSTMpreprocessing"),
        }
        lstm_model_params = {
            "epochs": get_value("epochs", "LSTM_model_params"),
            "batch_size": get_value("batch_size", "LSTM_model_params"),
            "learning_rate": get_value("learning_rate", "LSTM_model_params"),
            "metrics": get_value("metrics", "LSTM_model_params"),
            "dropout": get_value("dropout", "LSTM_model_params"),
        }
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=3,
            verbose=1,
            min_delta=0.01,
            restore_best_weights=True,
        )

        print(lstm_preprocessing)
        print(lstm_model_params)
        print(dataset)
        wandb.config.update(lstm_preprocessing)
        wandb.config.update({"dataset": dataset})
        wandb.config.update(lstm_model_params)

        x_test, y_test, x_train, y_train = preprocessing_lstm(data, **lstm_preprocessing)
        n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(
                    64, input_shape=(n_timesteps, n_features), return_sequences=True
                ),
                tf.keras.layers.Dropout(lstm_model_params["dropout"]),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(lstm_model_params["dropout"]),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lstm_model_params["learning_rate"]
            ),
            loss=tf.keras.losses.Huber(),
            metrics=lstm_model_params["metrics"],
        )

        model.fit(
            x_train,
            y_train,
            epochs=lstm_model_params["epochs"],
            batch_size=lstm_model_params["batch_size"],
            validation_data=(x_test, y_test),
            callbacks=[WandbCallbackKeras(), early_stopping],
        )
        evaluation = model.evaluate(x_test, y_test)
        print(f"Evaluation Loss: {evaluation}")
        path = get_path()
        model.save(path)
        mse = evaluate_lstm_model(model=model, givenPreprocessingParam=lstm_preprocessing)
        wandb.log({"mse": mse})
        return path


def train_ard_regressor():
    print(param)
    data = get_data(param["dataset"])
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
        data, **preprocessing_param
    )

    model = BayesianRidge(n_iter=1000)
    model.fit(x_train, y_train)
    path = get_path()
    joblib.dump(
        model, os.path.join(path)
    )  # Speichern des Modells als ard_model.pkl im angegebenen Verzeichnis

    return path


def trainARDRegressorHP():
    print(param)
    data = get_data(param["dataset"])
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
        data, **preprocessing_param
    )

    model = ARDRegression()
    param_grid = {
        "n_iter": [100, 300, 500, 1000, 1500],
        "tol": [1e-3, 1e-4, 1e-5],
        "alpha_1": [1e-6, 1e-7, 1e-8],
        "alpha_2": [1e-6, 1e-7, 1e-8],
        "lambda_1": [1e-6, 1e-7, 1e-8],
        "lambda_2": [1e-6, 1e-7, 1e-8],
        "compute_score": [True, False],
    }
    grid_search = HalvingGridSearchCV(
        model,
        param_grid,
        cv=5,
        verbose=2,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
    )
    grid_search.fit(x_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    path = get_path()
    joblib.dump(best_model, os.path.join(path))

    return path


def train_k_neighbors_regressor():
    print(param)
    data = get_data(param["dataset"])
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
        data, **preprocessing_param
    )

    model = KNeighborsRegressor(
        n_neighbors=100, weights="distance", algorithm="kd_tree", p=1, leaf_size=60
    )
    model.fit(x_train, y_train)
    path = get_path()
    joblib.dump(model, os.path.join(path))

    return path


def train_k_neighbors_regressor_hp():
    print(param)
    data = get_data(param["dataset"])
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
        data, **preprocessing_param
    )

    model = KNeighborsRegressor()

    param_grid = {
        "n_neighbors": [20, 25, 50, 75, 100, 200],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2],
        "leaf_size": [15, 30, 60, 90],
    }

    grid_search = HalvingGridSearchCV(
        model,
        param_grid,
        cv=5,
        verbose=2,
        scoring=make_scorer(
            mean_squared_error, greater_is_better=False
        ),
    )

    grid_search.fit(x_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    path = get_path()
    joblib.dump(
        best_model, os.path.join(path)
    )  # Speichern des Modells als ard_model.pkl im angegebenen Verzeichnis

    return path


if __name__ == "__main__":
    wandb.agent(getSweepIDXGBoost(), trainXGBRegressor)
    # wandb.agent(getSweepIDXGBoost(), trainXGBRegressor)
    # trainLSTM()
    # trainXGBRegressor()
    # evaluateLSTMModel("..\models\model_2023-10-17_21-43-29.json")
