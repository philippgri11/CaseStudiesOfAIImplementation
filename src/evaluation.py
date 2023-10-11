import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from src.loadData import getData

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)

def evaluateDNNModel(path):
    loaded_model = tf.keras.models.load_model(path)
    xTest, yTest, xTrain, yTrain = getData(param['dataset'], param['test_size'])
    predictedLabels = loaded_model.predict(xTest)
    evaluateModel(predictedLabels)

def evaluateXGBModel(path):
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model(path)
    xTest, yTest, xTrain, yTrain = getData(param['dataset'], param['test_size'])
    predictedLabels = loaded_model.predict(xTest)
    evaluateModel(predictedLabels)

def evaluateModel(predictedLabels):
    xTest, yTest, xTrain, yTrain = getData(param['dataset'], param['test_size'])
    mse = mean_squared_error(yTest.to_numpy(), predictedLabels)
    print(f'MSE is {mse}')
    plot(yTest,predictedLabels)


def plot(yTest,predictedLabels ):
    plt.figure(figsize=(8, 6))
    plt.plot(yTest.to_numpy()[0:1800], label='real Values (yTest)', marker='o')
    plt.plot(predictedLabels[0:1800], label='predicted Values', marker='x')
    plt.legend()
    plt.title('Predicted values vs. actual values')
    plt.grid(True)
    plt.show()