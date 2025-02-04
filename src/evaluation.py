import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

import wandb
from src.preprocessing import preprocessing, preprocessing_lstm
from src.load_data import get_data


def get_param():
    with open("../params.yaml", "r") as file:
        return yaml.safe_load(file)
        defaultPreprocessingParam = param["preprocessing"]
        lstm_preprocessing = param["LSTMpreprocessing"]


def get_default_preprocessing_param():
    with open("../params.yaml", "r") as file:
        param = yaml.safe_load(file)
        return param["preprocessing"]


def get_lstm_preprocessing():
    with open("../params.yaml", "r") as file:
        param = yaml.safe_load(file)
        return param["LSTMpreprocessing"]


def evaluate_sklearn_model(path=None, model=None, given_preprocessing_param=None):
    """
    Evaluates an sklearn Regression model using test data.

    Parameters
    ----------
    path : str
        The path to the ARD Regression model.
    model : xgb.XGBRegressor, optional
        An already loaded XGBoost model, if available.
    given_preprocessing_param : dict, optional
        Preprocessing parameters to override the defaults.

    Notes
    -----
    This function loads the sklearn model, preprocesses the data using XGBoost preprocessing,
    predicts labels for the test data, and evaluates the model using predefined metrics.
    """
    if model is None:
        model = joblib.load(path)
    if given_preprocessing_param is not None:
        preprocessing_param = given_preprocessing_param
    else:
        param = get_param()
        preprocessing_param = get_default_preprocessing_param()
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
        get_data(get_param()["dataset"]), **preprocessing_param
    )
    predicted_labels_test = model.predict(x_test)
    predicted_labels_val = model.predict(x_val)
    return evaluate_model(predicted_labels_test, y_test, predicted_labels_val, y_val)


def evaluate_dnn_model(path):
    """
    Evaluates a Deep Neural Network (DNN) model using test data.

    Parameters
    ----------
    path : str
        The path to the serialized DNN model.

    Returns
    -------
    float
        The mean squared error value on the validation data.

    Notes
    -----
    This function loads the DNN model, preprocesses the data using XGBoost preprocessing,
    predicts labels for the test data, and evaluates the model using predefined metrics.
    """
    loaded_model = tf.keras.models.load_model(path)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
        get_data(get_param()["dataset"]), **get_default_preprocessing_param()
    )
    predictedLabels = loaded_model.predict(x_test)
    return evaluate_model(predictedLabels, y_test)


def evaluate_lstm_model(
    path=None,
    model=None,
    givenPreprocessingParam=None,
    x_val_given=None,
    y_val_given=None,
    x_test_given=None,
    y_test_given=None,
):
    """
    Evaluates an LSTM model using test data, with optional model loading and preprocessing parameters.

    Parameters
    ----------
    path : str, optional
        The path to the serialized LSTM model, if the model is not provided directly.
    model : tf.keras.Model, optional
        An already loaded LSTM model, if available.
    givenPreprocessingParam : dict, optional
        Preprocessing parameters to override the defaults.

    Returns
    -------
    float
        The mean squared error value on the validation data.
    """
    if model is None:
        model = tf.keras.models.load_model(path)
    data = get_data(get_param()["dataset"])
    preprocessing_params = givenPreprocessingParam or get_lstm_preprocessing()
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_lstm(
        data, **preprocessing_params
    )
    if x_val_given is not None and y_val_given is not None:
        x_val = x_val_given
        y_val = y_val_given
    if x_test_given is not None and y_test_given is not None:
        x_test = x_test_given
        y_test = y_test_given
    predicted_labels_val = model.predict(x_val)
    predicted_labels_test = model.predict(x_test)
    return evaluate_model(predicted_labels_test, y_test, predicted_labels_val, y_val)


def write_results_to_csv(
    initial_data, path=None, model=None, given_preprocessing_param=None
):
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model(path)
    preprocessingParams = given_preprocessing_param or get_default_preprocessing_param()
    data = preprocessing(get_data(get_param()["dataset"]), **preprocessingParams)
    lag_features = initial_data[-given_preprocessing_param["lag"] :].tolist()

    predictions = []

    for i in range(len(data)):
        # Führe die Vorhersage mit den aktuellen Lag-Features durch
        row = data[i]
        row[f"electricLoad_rolling_avg_{given_preprocessing_param['lag']}"] = np.mean(
            lag_features
        )
        prediction = model.predict(row)

        # Füge die Vorhersage zur Liste der Vorhersagewerte hinzu
        predictions.append(prediction)

        # Aktualisiere die Lag-Features: Entferne den ältesten Wert und füge die neue Vorhersage hinzu
        lag_features.pop(0)
        lag_features.append(prediction)

    return predictions


def evaluate_xgb_model(path=None, model=None, given_preprocessing_param=None):
    """
    Evaluates an XGBoost model using test and validation data, with optional model loading and preprocessing parameters.

    Parameters
    ----------
    path : str, optional
        The path to the serialized XGBoost model, if the model is not provided directly.
    model : xgb.XGBRegressor, optional
        An already loaded XGBoost model, if available.
    given_preprocessing_param : dict, optional
        Preprocessing parameters to override the defaults.

    Returns
    -------
    float
        The mean squared error value on the validation data.
    """
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model(path)
    preprocessingParams = given_preprocessing_param or get_default_preprocessing_param()
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing(
        get_data(get_param()["dataset"]), **preprocessingParams
    )
    predictedLabelsTest = model.predict(xTest)
    predictedLabelsVal = model.predict(xVal)
    return evaluate_model(predictedLabelsTest, yTest, predictedLabelsVal, yVal)


def evaluate_model(
    predictedLabelsTest, y_test=None, predictedLabelsVal=None, y_val=None
):
    """
    Evaluates model predictions against actual labels using multiple metrics.

    Parameters
    ----------
    predictedLabelsTest : array-like
        Predicted labels for the test dataset.
    y_test : pd.Series or array-like, optional
        Actual labels for the test dataset.
    predictedLabelsVal : array-like, optional
        Predicted labels for the validation dataset.
    y_val : pd.Series or array-like, optional
        Actual labels for the validation dataset.

    Returns
    -------
    float
        The mean squared error value on the validation data.

    Notes
    -----
    Logs the evaluation metrics to Weights & Biases (wandb) for visualization and tracking.
    """
    if y_test is None:
        # This is a placeholder for your data preprocessing function
        x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
            get_data(get_param()["dataset"]), **get_default_preprocessing_param()
        )

    # Calculate metrics
    residuals_test = np.asarray(y_test) - predictedLabelsTest
    residuals_val = (
        np.asarray(y_val) - predictedLabelsVal
        if predictedLabelsVal is not None
        else None
    )

    mse_test = mean_squared_error(np.asarray(y_test), predictedLabelsTest)
    mse_val = (
        mean_squared_error(np.asarray(y_val), predictedLabelsVal)
        if predictedLabelsVal is not None
        else None
    )
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(np.asarray(y_test), predictedLabelsTest)
    r2_test = r2_score(np.asarray(y_test), predictedLabelsTest)

    variance_residuals_test = np.var(residuals_test)
    variance_residuals_val = (
        np.var(residuals_val) if residuals_val is not None else None
    )

    # Log metrics
    wandb.log(
        {
            "mse_test": mse_test,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test,
            "variance_residuals_test": variance_residuals_test,
            "residuals_test_histogram": wandb.Histogram(residuals_test),
        }
    )

    if residuals_val is not None:
        wandb.log(
            {
                "mse_val": mse_val,
                "variance_residuals_val": variance_residuals_val,
                "residuals_val_histogram": wandb.Histogram(residuals_val),
            }
        )

    return mse_val


def plot(yTest, predictedLabels):
    """
    Plots the actual vs. predicted labels for a subset of the test dataset.

    Parameters
    ----------
    yTest : pd.Series or array-like
        Actual labels for the test dataset.
    predictedLabels : array-like
        Predicted labels for the test dataset.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(yTest.to_numpy()[0:1800], label="real Values (yTest)", marker="o")
    plt.plot(predictedLabels[0:1800], label="predicted Values", marker="x")
    plt.legend()
    plt.title("Predicted values vs. actual values")
    plt.grid(True)
    plt.show()
