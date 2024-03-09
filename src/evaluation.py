import wandb
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import get_data
from preprocessing import preprocessing, preprocessing_lstm
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


with open("../params.yaml", "r") as file:
    param = yaml.safe_load(file)
    defaultPreprocessingParam = param["preprocessing"]
    lstm_preprocessing = param["LSTMpreprocessing"]




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
        preprocessing_param = defaultPreprocessingParam
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing(
        get_data(param["dataset"]), **preprocessing_param
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
        get_data(param["dataset"]), **defaultPreprocessingParam
    )
    predictedLabels = loaded_model.predict(x_test)
    return evaluate_model(predictedLabels, y_test)


def evaluate_lstm_model(path=None, model=None, givenPreprocessingParam=None):
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
    data = get_data(param["dataset"])
    preprocessing_params = givenPreprocessingParam or lstm_preprocessing
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_lstm(
        data, **preprocessing_params
    )
    predicted_labels = model.predict(x_test)
    return evaluate_model(predicted_labels, y_test)


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
    preprocessingParams = given_preprocessing_param or defaultPreprocessingParam
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing(
        get_data(param["dataset"]), **preprocessingParams
    )
    predictedLabelsTest = model.predict(xTest)
    predictedLabelsVal = model.predict(xVal)
    return evaluate_model(predictedLabelsTest, yTest, predictedLabelsVal, yVal)


def evaluate_model(predictedLabelsTest, y_test=None, predictedLabelsVal=None, y_val=None):
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
            get_data(param["dataset"]), **defaultPreprocessingParam
        )

    # Calculate metrics
    residuals_test = y_test.to_numpy() - predictedLabelsTest
    residuals_val = y_val.to_numpy() - predictedLabelsVal if predictedLabelsVal is not None else None

    mse_test = mean_squared_error(y_test.to_numpy(), predictedLabelsTest)
    mse_val = mean_squared_error(y_val.to_numpy(), predictedLabelsVal) if predictedLabelsVal is not None else None
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test.to_numpy(), predictedLabelsTest)
    r2_test = r2_score(y_test.to_numpy(), predictedLabelsTest)

    variance_residuals_test = np.var(residuals_test)
    variance_residuals_val = np.var(residuals_val) if residuals_val is not None else None

    # Log metrics
    wandb.log({
        "mse_test": mse_test,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "variance_residuals_test": variance_residuals_test,
        "residuals_test_histogram": wandb.Histogram(residuals_test)
    })

    if residuals_val is not None:
        wandb.log({
            "mse_val": mse_val,
            "variance_residuals_val": variance_residuals_val,
            "residuals_val_histogram": wandb.Histogram(residuals_val)
        })

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



