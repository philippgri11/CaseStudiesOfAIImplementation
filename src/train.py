import os
from datetime import datetime

import pandas as pd
from sklearn.experimental import enable_halving_search_cv
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
from evaluation import evaluate_xgb_model, evaluate_lstm_model, evaluate_sklearn_model
from load_data import get_data
from preprocessing import preprocessing, preprocessing_lstm
from sweep_config import getSweepIDLSTM, get_sweep_ard, get_sweep_k_neighbors, get_sweep_id_xg_boost

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
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
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
        )
        # save
        path = get_path()
        model.save_model(path)
        evaluate_xgb_model(model=model, given_preprocessing_param=preprocessing_param)
        return path

def trainXGBRegressorWithCV():
    """
    Trains an XGBoost regressor model with Cross-Validation to determine the best hyperparameters,
    evaluates the model, and saves it.

    Returns
    -------
    str
        The path where the trained model is saved.
    """
    with wandb.init(project="CaseStudiesOfAIImplementation", entity="philippgrill") as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
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

        dtrain_full = xgb.DMatrix(pd.concat([x_train, x_val]), label=pd.concat([y_train, y_val]))

        cv_results = xgb.cv(
            xgb_regressor_model_params,
            dtrain_full,
            num_boost_round=get_value("n_estimators", "XGBRegressorModelParams"),
            nfold=5,
            metrics='mphe',
            early_stopping_rounds=10,
            stratified=False,
            seed=42,
            callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)]
        )
        for metric in cv_results:
            for fold in range(len(cv_results[metric])):
                wandb.log({f"{metric}_fold_{fold}": cv_results[metric][fold]})

        # Now train the final model on the full dataset (train + validation)
        model = xgb.XGBRegressor(**xgb_regressor_model_params,n_estimators=get_value("n_estimators", "XGBRegressorModelParams"))

        model.fit(
            x_train,
            y_train,
            verbose=True,
            eval_set=[(x_val, y_val)],
            callbacks=[WandbCallback()],
        )

        # Save the model
        path = get_path()
        model.save_model(path)

        # Evaluate the final model
        evaluate_xgb_model(model=model, given_preprocessing_param=preprocessing_param)
        return path

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
        project="CaseStudiesOfAIImplementation", entity="philippgrill"
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


def train_lstm():
    """
        Trains a Long Short-Term Memory (LSTM) model, evaluates the model, logs the evaluation metrics to Weights & Biases, and saves the model.

        Returns
        -------
        str
            The path where the trained LSTM model is saved.
    """
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
            "monthly_cols": get_value("monthly_cols", "LSTMpreprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "LSTMpreprocessing"),
            "daily_cols": get_value("dailyCols", "LSTMpreprocessing"),
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

        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_lstm(data, **lstm_preprocessing)
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
            optimizer=tf.keras.optimizers.legacy.Adam(
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
    """
        Trains an Automatic Relevance Determination (ARD) regressor model, and saves it.

        Returns
        -------
        str
            The path where the trained ARD regressor model is saved.
    """
    with wandb.init(
        project="CaseStudiesOfAIImplementation", entity="philippgrill"
    ) as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        dataset = wandb.config.get("dataset") or param.get("dataset")
        data = get_data(dataset)
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "monthly_cols": get_value("monthly_cols", "preprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "preprocessing"),
            "daily_cols": get_value("daily_cols", "preprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "preprocessing"),
            "load_lag": get_value("load_lag", "preprocessing"),
        }

        ard_param = {
            "max_iter": get_value("max_iter", "ard"),
            "tol": get_value("tol", "ard"),
            "alpha_1": get_value("alpha_1", "ard"),
            "alpha_2": get_value("alpha_2", "ard"),
            "lambda_1": get_value("lambda_1", "ard"),
            "lambda_2": get_value("lambda_2", "ard"),
            "compute_score": get_value("compute_score", "ard")
        }
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
            data, **preprocessing_param
        )

        model = ARDRegression(**ard_param)
        model.fit(x_train, y_train)
        evaluate_sklearn_model(model=model, given_preprocessing_param=preprocessing_param)
        path = get_path()
        joblib.dump(
            model, os.path.join(path)
        )  # Speichern des Modells als ard_model.pkl im angegebenen Verzeichnis

    return path


def train_k_neighbors_regressor():
    """
        Trains a K-Neighbors regressor model, and saves it.

        Returns
        -------
        str
            The path where the trained K-Neighbors regressor model is saved.
    """
    with wandb.init(
        project="CaseStudiesOfAIImplementation", entity="philippgrill"
    ) as run:
        now = datetime.now()
        run.name = now.strftime("%d%m%Y%H%M%S")
        dataset = wandb.config.get("dataset") or param.get("dataset")
        data = get_data(dataset)
        preprocessing_param = {
            "test_size": get_value("test_size", "preprocessing"),
            "val_size": get_value("val_size", "preprocessing"),
            "columns": get_value("columns", "preprocessing"),
            "shifts": get_value("shifts", "preprocessing"),
            "neg_shifts": get_value("neg_shifts", "preprocessing"),
            "enable_daytime_index": get_value("enable_daytime_index", "preprocessing"),
            "monthly_cols": get_value("monthly_cols", "preprocessing"),
            "keep_monthly_avg": get_value("keep_monthly_avg", "preprocessing"),
            "daily_cols": get_value("daily_cols", "preprocessing"),
            "keep_daily_avg": get_value("keep_daily_avg", "preprocessing"),
            "load_lag": get_value("load_lag", "preprocessing"),
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

        model = KNeighborsRegressor(
            **k_neighbors_param
        )
        model.fit(x_train, y_train)
        evaluate_sklearn_model(model=model, given_preprocessing_param=preprocessing_param)

        path = get_path()
        joblib.dump(model, os.path.join(path))

    return path


if __name__ == "__main__":
    # wandb.agent(get_sweep_k_neighbors(), train_k_neighbors_regressor)
    # wandb.agent(get_sweep_ard(), train_ard_regressor)
    # wandb.agent(get_sweep_id_xg_boost(), trainXGBRegressorWithCV)
    # wandb.agent(getSweepIDLSTM(), train_lstm)
    wandb.agent(get_sweep_id_xg_boost(), trainXGBRegressor)
    # trainLSTM()
    # trainXGBRegressor()
    # evaluateLSTMModel("..\models\model_2023-10-17_21-43-29.json")
