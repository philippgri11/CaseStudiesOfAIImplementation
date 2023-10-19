import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def slidingWindow(df, colums, shifts, negShifts):
    if (shifts > 0):
        for shift in range(1, shifts + 1):
            for col in np.asarray(colums).astype('str'):
                df[col + str(shift)] = df[col].shift(shift)
    if negShifts < 0:
        for negShift in range(negShifts, 0):
            for col in colums:
                df[col + str(negShift)] = df[col].shift(negShift)
    df = df.dropna()
    return df


def prepareDataLSTM(df, shifts):
    data = []
    for i in range(len(df) - shifts + 1):
        seq = df.iloc[i:i + shifts].values  # Hole die nächsten n_steps Zeilen
        data.append(seq)
    return np.array(data)


def preprocessingLSTM(data,  shifts, test_size=0.2, target='electricLoad'):
    xTest, yTest, xTrain, yTrain = splitData(data, target, test_size=test_size)
    xTestScaler = StandardScaler()
    xTrainScaler = StandardScaler()
    xTest = pd.DataFrame(xTestScaler.fit_transform(xTest.to_numpy()))
    xTrain = pd.DataFrame(xTrainScaler.fit_transform(xTrain.to_numpy()))
    return prepareDataLSTM(xTest, shifts), yTest.iloc[:-shifts+1], prepareDataLSTM(xTrain,shifts), yTrain.iloc[:-shifts+1]


def splitData(data, target, test_size=0.2):
    train_df = data.head(int(len(data) * (1 - test_size)))
    test_df = data.tail(int(len(data) * (test_size)))
    xTest = test_df.drop(target, axis=1)
    yTest = test_df[target]
    xTrain = train_df.drop(target, axis=1)
    yTrain = train_df[target]
    return xTest, yTest, xTrain, yTrain

def preprocessing(data, test_size, colums, shifts, negShifts):
    data = slidingWindow(data, colums, shifts, negShifts)
    data = splitData(data, 'electricLoad', test_size)
    return data
