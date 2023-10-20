import wandb
import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from src.loadData import getData
from src.preprocessing import preprocessingXGBoost, preprocessingLSTM
import joblib

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)
    defaultPreprocessingParam = param['preprocessing']
    LSTMpreprocessing = param['LSTMpreprocessing']

def evaluateKNeighbors(path):
    loaded_model = joblib.load(path)
    xTest, yTest, xTrain, yTrain = preprocessingXGBoost(getData(param['dataset']), **defaultPreprocessingParam)
    predictedLabels = loaded_model.predict(xTest)
    evaluateModel(predictedLabels)

def evaluateARDModel(path):
    loaded_model = joblib.load(path)
    xTest, yTest, xTrain, yTrain = preprocessingXGBoost(getData(param['dataset']), **defaultPreprocessingParam)
    predictedLabels = loaded_model.predict(xTest)
    evaluateModel(predictedLabels)

def evaluateDNNModel(path):
    loaded_model = tf.keras.models.load_model(path)
    xTest, yTest, xTrain, yTrain = preprocessingXGBoost(getData(param['dataset']), **defaultPreprocessingParam)
    predictedLabels = loaded_model.predict(xTest)
    return evaluateModel(predictedLabels, yTest)

def evaluateLSTMModel(path= None, model = None, givenPreprocessingParam = None):
    if model is None:
        model = tf.keras.models.load_model(path)
    data = getData(param['dataset'])
    preprocessingParams =  givenPreprocessingParam or LSTMpreprocessing
    xTest, yTest, xTrain, yTrain = preprocessingLSTM(data, **preprocessingParams)
    predictedLabels = model.predict(xTest)
    return evaluateModel(predictedLabels, yTest)

def evaluateXGBModel(path= None, model = None, givenPreprocessingParam = None):
    if model is None:
        model = xgb.XGBRegressor()
        model.load_model(path)
    preprocessingParams =  givenPreprocessingParam or defaultPreprocessingParam
    xTest, yTest, xTrain, yTrain = preprocessingXGBoost(getData(param['dataset']), **preprocessingParams)
    predictedLabels = model.predict(xTest)
    return evaluateModel(predictedLabels, yTest)

def evaluateModel(predictedLabels, yTest =None):
    if yTest is None:
        xTest, yTest, xTrain, yTrain = preprocessingXGBoost(getData(param['dataset']), **defaultPreprocessingParam)
    mse = mean_squared_error(yTest.to_numpy(), predictedLabels)
    for i in range(len(predictedLabels)):
        wandb.log({"predictedLabels": predictedLabels[i]})
        wandb.log({"yTest": yTest.to_numpy()[i]})

    print(f'MSE is {mse}')
    # plot(yTest,predictedLabels)
    return mse


def plot(yTest,predictedLabels ):
    plt.figure(figsize=(8, 6))
    plt.plot(yTest.to_numpy()[0:1800], label='real Values (yTest)', marker='o')
    plt.plot(predictedLabels[0:1800], label='predicted Values', marker='x')
    plt.legend()
    plt.title('Predicted values vs. actual values')
    plt.grid(True)
    plt.show()