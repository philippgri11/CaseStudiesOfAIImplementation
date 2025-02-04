import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import yaml
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    HalvingGridSearchCV,
    cross_val_score,
    KFold,
    cross_validate,
)
from sklearn.neighbors import KNeighborsRegressor
from wandb import sweep
from wandb.integration.xgboost import WandbCallback
from wandb.keras import WandbCallback as WandbCallbackKeras

import wandb
from src.evaluation import (
    evaluate_lstm_model,
    evaluate_sklearn_model,
    evaluate_xgb_model,
)
from src.load_data import get_data
from src.preprocessing import preprocessing, preprocessing_lstm
from src.sweep_config import (
    get_sweep_ard,
    get_sweep_id_xg_boost,
    get_sweep_k_neighbors,
    getSweepIDLSTM,
)

max_runs = 1

with open("../params.yaml", "r") as file:
    param = yaml.safe_load(file)
    xgb_regressor_model_params = param["XGBRegressorModelParams"]
    dnn_model_params = param["DNN_model_params"]
    preprocessing_param = param["preprocessing"]
    lstm_preprocessing = param["LSTMpreprocessing"]
    lstm_model_params = param["LSTM_model_params"]


def get_value(key, group):
    """
    Retrieves a value for a given key from Weights & Biases configuration or fallbacks to parameters defined in YAML.

    Parameters
    ----------
    key : str
        The key for the desired parameter.
    group : str
        The parameter group within the YAML file.

    Returns
    -------
    value
        The retrieved value for the given key.
    """
    if wandb.config.get(key) is not None:
        return wandb.config.get(key)
    return param.get(group).get(key)


def trainXGBRegressor():
    """
    Trains an XGBoost regressor model with parameters defined in Weights & Biases or YAML file, evaluates the model, and saves it.

    Returns
    -------
    str
        The path where the trained model is saved.
    """
    project = "CaseStudiesOfAIImplementationResults"
    entity = "philippgrill"
    with wandb.init(
        project="CaseStudiesOfAIImplementationResults", entity="philippgrill"
    ) as run:
        if setup_sweep(entity, project, run):
            return

        with open("../params.yaml", "r") as file:
            param = yaml.safe_load(file)
        dataset = wandb.config.get("dataset") or param.get("dataset")
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "enable_day_of_week_index": get_value(
                "enable_day_of_week_index", "preprocessing"
            ),
            "monthly_cols": get_value("monthly_cols", "preprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "preprocessing"),
            "daily_cols": get_value("daily_cols", "preprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "preprocessing"),
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
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
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
            early_stopping_rounds=10,
        )
        # save
        path = get_path()
        model.save_model(path)
        evaluate_xgb_model(model=model, given_preprocessing_param=preprocessing_param)
        return path


def setup_sweep(entity, project, run):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{run.sweep_id}")
    now = datetime.now()
    run.name = now.strftime("%d%m%Y%H%M%S")
    if len(sweep.runs) > max_runs:
        command = f"wandb sweep --stop {run.sweep_id}"
        os.system(command)
        return True
    return False


def trainXGBRegressorWithCV():
    """
    Trains an XGBoost regressor model with Cross-Validation to determine the best hyperparameters,
    evaluates the model, and saves it.

    Returns
    -------
    str
        The path where the trained model is saved.
    """
    entity = "philippgrill"
    project = "CaseStudiesOfAIImplementationResults"
    with wandb.init(project=project, entity=entity) as run:
        if setup_sweep(entity, project, run):
            return

        with open("../params.yaml", "r") as file:
            param = yaml.safe_load(file)

        dataset = wandb.config.get("dataset") or param.get("dataset")
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
            "enable_day_of_week_index": get_value(
                "enable_day_of_week_index", "preprocessing"
            ),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "monthly_cols": get_value("monthly_cols", "preprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "preprocessing"),
            "daily_cols": get_value("daily_cols", "preprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "preprocessing"),
            "load_lag": get_value("load_lag", "preprocessing"),
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
            "tree_method": get_value("tree_method", "XGBRegressorModelParams"),
            "eval_metric": get_value("eval_metric", "XGBRegressorModelParams"),
        }
        wandb.config.update(xgb_regressor_model_params)
        wandb.config.update({"dataset": dataset})
        wandb.config.update(preprocessing_param)

        data = get_data(dataset)
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
            data, **preprocessing_param
        )

        dtrain_full = xgb.DMatrix(
            pd.concat([x_train, x_val]), label=pd.concat([y_train, y_val])
        )

        cv_results = xgb.cv(
            xgb_regressor_model_params,
            dtrain_full,
            num_boost_round=get_value("n_estimators", "XGBRegressorModelParams"),
            nfold=5,
            metrics="rmse",
            early_stopping_rounds=10,
            stratified=False,
            seed=42,
            # callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)],
        )
        for metric in cv_results:
            for fold_index, value in enumerate(cv_results[metric]):
                wandb.log({metric: value})

        wandb.log(
            {
                "mse_cv": cv_results["test-rmse-mean"].tail(1).item()
                * cv_results["test-rmse-mean"].tail(1).item(),
            }
        )


def get_path():
    """
    Generates a file path for saving a model with a timestamp.

    Returns
    -------
    str
        The file path where a model should be saved.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"model_{timestamp}.json"
    model_dir = "../models"
    path = os.path.join(model_dir, filename)
    print(f"saved to {path}")
    return path


def train_dnn():
    """
    Trains a Deep Neural Network (DNN) model using TensorFlow and Keras, evaluates the model, and saves it.

    Returns
    -------
    str
        The path where the trained DNN model is saved.
    """
    with wandb.init(
        project="CaseStudiesOfAIImplementationResults", entity="philippgrill"
    ) as run:
        data = get_data(param["dataset"])
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
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


def train_lstm(cv=False):
    """
    Trains a Long Short-Term Memory (LSTM) model, evaluates the model, logs the evaluation metrics to Weights & Biases, and saves the model.

    Returns
    -------
    str
        The path where the trained LSTM model is saved.
    """
    entity = "philippgrill"
    project = "CaseStudiesOfAIImplementationResults"
    with wandb.init(project=project, entity=entity) as run:
        if setup_sweep(entity, project, run):
            return
        dataset = wandb.config.get("dataset") or param.get("dataset")
        data = get_data(dataset)
        lstm_preprocessing = {
            "test_size": get_value("test_size", "LSTMpreprocessing"),
            "shifts": get_value("shifts", "LSTMpreprocessing"),
            "enable_daytime_index": get_value(
                "enable_daytime_index", "LSTMpreprocessing"
            ),
            "enable_day_of_week_index": get_value(
                "enable_day_of_week_index", "preprocessing"
            ),
            "monthly_cols": get_value("monthly_cols", "LSTMpreprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "LSTMpreprocessing"),
            "daily_cols": get_value("daily_cols", "LSTMpreprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "LSTMpreprocessing"),
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
            patience=6,
            verbose=1,
            min_delta=0.1,
            restore_best_weights=True,
        )

        print(lstm_preprocessing)
        print(lstm_model_params)
        print(dataset)
        wandb.config.update(lstm_preprocessing)
        wandb.config.update({"dataset": dataset})
        wandb.config.update(lstm_model_params)

        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_lstm(
            data, **lstm_preprocessing
        )
        n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
        if cv:
            n_splits = 3
            kf = KFold(n_splits=n_splits, shuffle=False)

            fold = 0
            x_train_full = np.concatenate([x_train, x_val], axis=0)
            y_train_full = np.concatenate([y_train, y_val], axis=0)
            mse = 0
            for train_index, val_index in kf.split(x_train_full, y_train_full):
                print(f"Training fold {fold + 1}/{n_splits}...")
                x_train_fold, x_val_fold = (
                    x_train_full[train_index],
                    x_train_full[val_index],
                )
                y_train_fold, y_val_fold = (
                    y_train_full[train_index],
                    y_train_full[val_index],
                )

                model = build_lstm_model(x_train.shape[1:], **lstm_model_params)

                model.fit(
                    x_train_fold,
                    y_train_fold,
                    validation_data=(x_val_fold, y_val_fold),
                    callbacks=[WandbCallbackKeras(), early_stopping],
                    epochs=lstm_model_params["epochs"],
                )
                mse += evaluate_lstm_model(
                    model=model,
                    givenPreprocessingParam=lstm_preprocessing,
                    x_val_given=x_val_fold,
                    y_val_given=y_val_fold.tolist(),
                )
                wandb.log({f"mse_fold_{fold + 1}/{n_splits}": mse})
                fold += 1
            wandb.log({"mse_cv": mse / n_splits})
            return None

        else:
            model = build_lstm_model((n_timesteps, n_features), **lstm_model_params)
            x_train_full = np.concatenate([x_train, x_val], axis=0)
            y_train_full = np.concatenate([y_train, y_val], axis=0)
            model.fit(
                x_train_full,
                y_train_full,
                epochs=lstm_model_params["epochs"],
                batch_size=lstm_model_params["batch_size"],
                validation_data=(x_test, y_test),
                callbacks=[WandbCallbackKeras(), early_stopping],
            )
            evaluation = model.evaluate(x_test, y_test)
            print(f"Evaluation Loss: {evaluation}")
            path = get_path()
            model.save(path)
            mse = evaluate_lstm_model(
                model=model, givenPreprocessingParam=lstm_preprocessing
            )
            wandb.log({"mse": mse})
            return path


def build_lstm_model(input_shape, **params):
    devices = tf.config.list_physical_devices()
    print("\nDevices: ", devices)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU details: ", details)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    512, input_shape=input_shape, return_sequences=True
                ),
                merge_mode="concat",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"]),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=True), merge_mode="concat"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params["learning_rate"],
        decay_steps=params["epochs"] / 2,
        decay_rate=0.5,
        staircase=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(),
        metrics=params["metrics"],
    )
    return model


def train_ard_regressor(cv=True):
    """
    Trains an Automatic Relevance Determination (ARD) regressor model, and saves it.

    Returns
    -------
    str
        The path where the trained ARD regressor model is saved.
    """
    entity = "philippgrill"
    project = "CaseStudiesOfAIImplementationResults"
    with wandb.init(project=project, entity=entity) as run:
        if setup_sweep(entity, project, run):
            return
        dataset = wandb.config.get("dataset") or param.get("dataset")
        data = get_data(dataset)
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
            "enable_day_of_week_index": get_value(
                "enable_day_of_week_index", "preprocessing"
            ),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "monthly_cols": get_value("monthly_cols", "preprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "preprocessing"),
            "daily_cols": get_value("daily_cols", "preprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "preprocessing"),
        }

        ard_param = {
            "max_iter": get_value("max_iter", "ard"),
            "tol": get_value("tol", "ard"),
            "alpha_1": get_value("alpha_1", "ard"),
            "alpha_2": get_value("alpha_2", "ard"),
            "lambda_1": get_value("lambda_1", "ard"),
            "lambda_2": get_value("lambda_2", "ard"),
            "compute_score": get_value("compute_score", "ard"),
        }
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
            data, **preprocessing_param
        )

        model = ARDRegression(**ard_param)
        if cv:
            cv_sklearn_model(x_train, y_train, x_val, y_val, model)
            return None
        else:
            model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
            evaluate_sklearn_model(
                model=model, given_preprocessing_param=preprocessing_param
            )
            path = get_path()
            joblib.dump(
                model, os.path.join(path)
            )  # Speichern des Modells als ard_model.pkl im angegebenen Verzeichnis
            return path


def train_k_neighbors_regressor(cv=True):
    """
    Trains a K-Neighbors regressor model, and saves it.

    Returns
    -------
    str
        The path where the trained K-Neighbors regressor model is saved.
    """
    entity = "philippgrill"
    project = "CaseStudiesOfAIImplementationResults"
    with wandb.init(project=project, entity=entity) as run:
        if setup_sweep(entity, project, run):
            return
        dataset = wandb.config.get("dataset") or param.get("dataset")
        data = get_data(dataset)
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
            "enable_day_of_week_index": get_value(
                "enable_day_of_week_index", "preprocessing"
            ),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "monthly_cols": get_value("monthly_cols", "preprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "preprocessing"),
            "daily_cols": get_value("daily_cols", "preprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "preprocessing"),
        }
        k_neighbors_param = {
            "n_neighbors": get_value("n_neighbors", "k_neighbors"),
            "p": get_value("p", "k_neighbors"),
            "algorithm": get_value("algorithm", "k_neighbors"),
            "leaf_size": get_value("leaf_size", "k_neighbors"),
            "weights": get_value("weights", "k_neighbors"),
        }
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
            data, **preprocessing_param
        )

        model = KNeighborsRegressor(**k_neighbors_param)
        if cv:
            cv_sklearn_model(x_train, y_train, x_val, y_val, model)
            return None
        else:
            model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
            evaluate_sklearn_model(
                model=model, given_preprocessing_param=preprocessing_param
            )

            path = get_path()
            joblib.dump(model, os.path.join(path))
            return path


def cv_sklearn_model(x_train, y_train, x_val, y_val, model):
    x_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    scores = cross_validate(
        model, x_train_full, y_train_full, cv=5, scoring="neg_mean_squared_error"
    )
    mean_score = -np.mean(scores["test_score"])
    wandb.log({"mse_cv": mean_score})
    for score in scores["test_score"]:
        wandb.log({"test_score": -score})


def switch_dataset(dataset: str):
    with open("../params.yaml", "r") as file:
        data = yaml.safe_load(file)
    data["dataset"] = dataset
    with open("../params.yaml", "w") as file:
        yaml.safe_dump(data, file)

    with open("../params.yaml", "r") as file:
        global param
        param = yaml.safe_load(file)
        global xgb_regressor_model_params
        xgb_regressor_model_params = param["XGBRegressorModelParams"]
        global dnn_model_params
        dnn_model_params = param["DNN_model_params"]
        global preprocessing_param
        preprocessing_param = param["preprocessing"]
        global lstm_preprocessing
        lstm_preprocessing = param["LSTMpreprocessing"]
        global lstm_model_params
        lstm_model_params = param["LSTM_model_params"]


if __name__ == "__main__":
    datasets = [
        "loadCurveOneFull",
        "loadCurveThreeFull",
        "loadCurveTwoFull",
    ]
    for dataset in datasets:
        switch_dataset(dataset)
        wandb.agent(get_sweep_k_neighbors(), train_k_neighbors_regressor)
        wandb.agent(get_sweep_ard(), train_ard_regressor)
        wandb.agent(getSweepIDLSTM(), train_lstm)
        wandb.agent(get_sweep_id_xg_boost(), trainXGBRegressorWithCV)
