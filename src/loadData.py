import pandas as pd

def readLoadCurveOnetToDataframe(file_path, sep=';', decimal=','):
    # The semicolon (;) is used as a separator and the comma (,) as a decimal separator.
    df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=0, names=['startDate', 'endDate', 'electricLoad'])
    df['startDate'] = pd.to_datetime(df['startDate'], format='%d.%m.%Y %H:%M')
    # df['endDate'] = pd.to_datetime(df['endDate'], format='%d.%m.%Y %H:%M')
    df = df.drop(columns='endDate').dropna()
    df = dateToFloat(df, ['startDate'])
    df = cleanData(df)
    return df.dropna()

def readLoadCurveTwoToDataframe(file_path, sep=';', decimal=','):
    # The semicolon (;) is used as a separator and the comma (,) as a decimal separator.
    df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=0, names=['startDate', 'endDate', 'electricLoad', 't1', 't2', 'r1', 'r2'])
    df['startDate'] = pd.to_datetime(df['startDate'], format='%d.%m.%Y %H:%M')
    # df['endDate'] = pd.to_datetime(df['endDate'], format='%d.%m.%Y %H:%M')
    df = df.drop(columns='endDate').dropna()
    df = dateToFloat(df, ['startDate'])
    df = cleanData(df)
    return df

def readLoadCurveThreeToDataframe(file_path, sep=';', decimal=','):
    # The semicolon (;) is used as a separator and the comma (,) as a decimal separator.
    df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=0,  names=['startDate', 'endDate', 'electricLoad', 't1',  'r1'])
    df['startDate'] = pd.to_datetime(df['startDate'], format='%d.%m.%Y %H:%M')
    # df['endDate'] = pd.to_datetime(df['endDate'], format='%d.%m.%Y %H:%M')
    df = df.drop(columns='endDate')
    df = dateToFloat(df, ['startDate'])
    df = cleanData(df)
    return df


def readHolidayToDataframe(file_path, sep=';'):
    # The semicolon (;) is used as a separator and the comma (,) as a decimal separator.
    df = pd.read_csv(file_path, sep=sep, header=0, names=['date', 'holiday', 'schoolHoliday'])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['date_year'] = df['date'].dt.year
    df['date_month'] = df['date'].dt.month
    df['date_day'] = df['date'].dt.day
    df.drop('date', axis=1, inplace=True)
    return df.dropna()

def dateToFloat(df, cols):
    # Split columns 'startDate' into numeric characteristics adds dayofweek
    for col in cols:
        df['dayofweek'] = df[col].dt.dayofweek
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_minute'] = df[col].dt.minute
    # Remove original date/time columns
    df.drop(cols, axis=1, inplace=True)
    return df

def cleanData(df):
    df.replace('#NA!', pd.NA, inplace=True)
    df.replace({',': '.'}, regex=True, inplace=True)
    df = df.dropna()
    df['electricLoad'] = df['electricLoad'].astype(float)
    df = df.dropna()
    return df

def mergeHolidayAndLoadCurve(holiday, loadCurveOne):
    merged_df = pd.merge(
        left=loadCurveOne,
        right=holiday,
        left_on=['startDate_year','startDate_month','startDate_day'],
        right_on=['date_year','date_month','date_day'],
        how='inner'
    )
    merged_df.drop('date_year', axis=1, inplace=True)
    merged_df.drop('date_month', axis=1, inplace=True)
    merged_df.drop('date_day', axis=1, inplace=True)

    return merged_df



def getData(dataset):
    if dataset == 'loadCurveOne':
        loadCurve = readLoadCurveOnetToDataframe("../data/training_data_period_1.csv")
    elif dataset == 'loadCurveTwo':
        loadCurve = readLoadCurveTwoToDataframe("../data/training_data_period_2.csv")
    elif dataset == 'loadCurveThree':
        loadCurve = readLoadCurveThreeToDataframe("../data/training_data_period_3.csv")
    else:
        raise ValueError("Datase Not Suported")
    holiday = readHolidayToDataframe("../data/holiday.csv")
    data = mergeHolidayAndLoadCurve(holiday, loadCurve)
    data['electricLoad'] = data['electricLoad'].astype(float)
    return data
