import numpy as np


def slidingWindow(df, colums, shifts, negShifts):
    if (shifts>0):
        for shift in range(1,shifts+1):
            for col in np.asarray(colums).astype('str'):
                df[col + str(shift)] = df[col].shift(shift)
    if negShifts<0:
        for negShift in range(negShifts,0):
            for col in colums:
                df[col + str(negShift)] = df[col].shift(negShift)
    df = df.dropna()
    return df


def splitData(data, target, test_size=0.2):
    train_df = data.head( int(len(data) * (1-test_size)))
    test_df = data.tail( int( len(data) * (test_size)))
    xTest = test_df.drop(target, axis=1)
    yTest = test_df[target]
    xTrain = train_df.drop(target, axis=1)
    yTrain = train_df[target]
    return xTest, yTest , xTrain , yTrain

def preprocessing(data, test_size, colums, shifts, negShifts):
    data = slidingWindow(data ,colums, shifts, negShifts)
    data = splitData(data, 'electricLoad', test_size)
    return data