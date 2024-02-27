import wandb
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import get_data
from preprocessing import preprocessing_xg_boost, preprocessing_lstm
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


with open("../params.yaml", "r") as file:
    param = yaml.safe_load(file)
    defaultPreprocessingParam = param["preprocessing"]
    lstm_preprocessing = param["LSTMpreprocessing"]


def evaluate_k_neighbors(path):
    """
        Evaluates a K-Nearest Neighbors model using test data.

        Parameters
        ----------
        path : str
            The path to the K-Nearest Neighbors model.

        Notes
        -----
        This function loads the model, preprocesses the data using XGBoost preprocessing,
        predicts labels for the test data, and evaluates the model using predefined metrics.
        """
    loaded_model = joblib.load(path)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
        get_data(param["dataset"]), **defaultPreprocessingParam
    )
    predicted_labels = loaded_model.predict(x_test)
    evaluate_model(predicted_labels)


def evaluate_ard_model(path):
    """
        Evaluates an ARD Regression model using test data.

        Parameters
        ----------
        path : str
            The path to the ARD Regression model.

        Notes
        -----
        This function loads the ARD model, preprocesses the data using XGBoost preprocessing,
        predicts labels for the test data, and evaluates the model using predefined metrics.
    """
    loaded_model = joblib.load(path)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
        get_data(param["dataset"]), **defaultPreprocessingParam
    )
    predicted_labels = loaded_model.predict(x_test)
    evaluate_model(predicted_labels)


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
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing_xg_boost(
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
    preprocessingParams = givenPreprocessingParam or lstm_preprocessing
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_lstm(
        data, **preprocessingParams
    )
    predictedLabels = model.predict(xTest)
    return evaluate_model(predictedLabels, yTest)


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
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_xg_boost(
        get_data(param["dataset"]), **preprocessingParams
    )
    predictedLabelsTest = model.predict(xTest)
    predictedLabelsVal = model.predict(xVal)
    return evaluate_model(predictedLabelsTest, yTest, predictedLabelsVal, yVal)


def evaluate_model(predictedLabelsTest, yTest=None, predictedLabelsVal=None, yVal=None):
    """
        Evaluates model predictions against actual labels using multiple metrics.

        Parameters
        ----------
        predictedLabelsTest : array-like
            Predicted labels for the test dataset.
        yTest : pd.Series or array-like, optional
            Actual labels for the test dataset.
        predictedLabelsVal : array-like, optional
            Predicted labels for the validation dataset.
        yVal : pd.Series or array-like, optional
            Actual labels for the validation dataset.

        Returns
        -------
        float
            The mean squared error value on the validation data.

        Notes
        -----
        Logs the evaluation metrics to Weights & Biases (wandb) for visualization and tracking.
        """
    if yTest is None:
        xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_xg_boost(
            get_data(param["dataset"]), **defaultPreprocessingParam
        )

    # Berechne die Metriken
    mse_test = mean_squared_error(yTest.to_numpy(), predictedLabelsTest)
    mse_val = mean_squared_error(yVal.to_numpy(), predictedLabelsVal)
    rmse_test = mean_squared_error(yTest.to_numpy(), predictedLabelsTest, squared=False)
    mae_test = mean_absolute_error(yTest.to_numpy(), predictedLabelsTest)
    r2_test = r2_score(yTest.to_numpy(), predictedLabelsTest)

    # Logge die Metriken
    for i in range(len(predictedLabelsTest)):
        wandb.log(
            {
                "predictedLabelsTest": predictedLabelsTest[i],
                "yTest": yTest.to_numpy()[i],
                "residual": yTest.to_numpy()[i]-predictedLabelsTest[i]
            }
        )

    for i in range(len(predictedLabelsVal)):
        wandb.log(
            {"predictedLabelsVal": predictedLabelsVal[i], "yVal": yVal.to_numpy()[i]}
        )

    wandb.log(
        {
            "mse_test": mse_test,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test,
            "mse_val": mse_val,
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


def evaluate_xgb_model_with_cv(
    path=None, model=None, given_preprocessing_param=None, cv_splits=5
):
    """
    Evaluates an XGBoost model using cross-validation.

    Parameters
    ----------
    path : str, optional
        The path to the serialized XGBoost model, if the model is not provided directly.
    model : xgb.XGBRegressor, optional
        An already loaded XGBoost model, if available.
    given_preprocessing_param : dict, optional
        Preprocessing parameters to override the defaults.
    cv_splits : int, optional
        The number of folds in K-Fold cross-validation.

    Returns
    -------
    tuple
        A tuple containing the average MSE, MAE, and R^2 values across all cross-validation splits.
    """
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model(path)

    preprocessingParams = given_preprocessing_param or defaultPreprocessingParam
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_xg_boost(
        get_data(param["dataset"]), **preprocessingParams
    )
    x_combined = np.concatenate((xTrain, xVal), axis=0)
    y_combined = np.concatenate((yTrain, yVal), axis=0)
    # Definiere KFold Cross-Validation
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Berechne Cross-Validation Metriken
    scores_mse = cross_val_score(
        model, x_combined, y_combined, cv=kfold, scoring="neg_mean_squared_error"
    )
    scores_mae = cross_val_score(
        model, x_combined, y_combined, cv=kfold, scoring="neg_mean_absolute_error"
    )
    scores_r2 = cross_val_score(model, x_combined, y_combined, cv=kfold, scoring="r2")

    # Negativen Werte umkehren, da 'neg_mean_squared_error' und 'neg_mean_absolute_error' negativ sind
    scores_mse = -scores_mse
    scores_mae = -scores_mae

    # Durchschnittliche Metriken
    avg_mse = np.mean(scores_mse)
    avg_mae = np.mean(scores_mae)
    avg_r2 = np.mean(scores_r2)

    # Logge die durchschnittlichen Metriken
    wandb.log({"avg_mse_cv": avg_mse, "avg_mae_cv": avg_mae, "avg_r2_cv": avg_r2})

    # Ergebnisse zurückgeben
    return avg_mse, avg_mae, avg_r2
