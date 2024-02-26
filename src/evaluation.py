import wandb
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from src.load_data import get_data
from src.preprocessing import preprocessing_xg_boost, preprocessing_lstm
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
    loaded_model = joblib.load(path)
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_xg_boost(
        get_data(param["dataset"]), **defaultPreprocessingParam
    )
    predictedLabels = loaded_model.predict(xTest)
    evaluate_model(predictedLabels)


def evaluate_ard_model(path):
    loaded_model = joblib.load(path)
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_xg_boost(
        get_data(param["dataset"]), **defaultPreprocessingParam
    )
    predictedLabels = loaded_model.predict(xTest)
    evaluate_model(predictedLabels)


def evaluate_dnn_model(path):
    loaded_model = tf.keras.models.load_model(path)
    xTrain, yTrain, xVal, yVal, xTest, yTest = preprocessing_xg_boost(
        get_data(param["dataset"]), **defaultPreprocessingParam
    )
    predictedLabels = loaded_model.predict(xTest)
    return evaluate_model(predictedLabels, yTest)


def evaluate_lstm_model(path=None, model=None, givenPreprocessingParam=None):
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
