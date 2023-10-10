import pandas as pd

def readCsvToDataframe(file_path, sep=';', decimal=','):
    # The semicolon (;) is used as a separator and the comma (,) as a decimal separator.
    df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=1, names=['startDate', 'endDate', 'electricLoad'])
    df['startDate'] = pd.to_datetime(df['startDate'], format='%d.%m.%Y %H:%M')
    df['endDate'] = pd.to_datetime(df['endDate'], format='%d.%m.%Y %H:%M')
    return df.dropna()

def dateToFloat(df):
    # Split columns 'startDate' and 'endDate' into numeric characteristics
    for col in ['startDate', 'endDate']:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_minute'] = df[col].dt.minute
        df[f'{col}_second'] = df[col].dt.second

    # Remove original date/time columns
    df.drop(['startDate', 'endDate'], axis=1, inplace=True)
    return df

def splitData(data, target, test_size):
    train_df = data.head( int(len(data) * (1-test_size)))
    test_df = data.tail( int( len(data) * (test_size)))
    xTest = test_df.drop(target, axis=1)
    yTest = test_df[target]
    xTrain = train_df.drop(target, axis=1)
    yTrain = train_df[target]
    return xTest, yTest , xTrain , yTrain