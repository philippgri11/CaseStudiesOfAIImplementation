import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pvlib
from datetime import timedelta, datetime
from astral.sun import sun
from astral import LocationInfo


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


def preprocessingLSTM(data, shifts, test_size=0.2, target='electricLoad', enable_daytime_index=True, dailyCols=None,
                      monthlyCols=None, keepDailyAvg=None, keepMonthlyAvg=None):
    data = commonPreprocessing(data, enable_daytime_index=enable_daytime_index, dailyCols=dailyCols,
                               monthlyCols=monthlyCols, keepDailyAvg=keepDailyAvg, keepMonthlyAvg=keepMonthlyAvg)
    xTest, yTest, xTrain, yTrain = splitData(data, target, test_size=test_size)
    xTestScaler = StandardScaler()
    xTrainScaler = StandardScaler()
    xTest = pd.DataFrame(xTestScaler.fit_transform(xTest.to_numpy()))
    xTrain = pd.DataFrame(xTrainScaler.fit_transform(xTrain.to_numpy()))
    return prepareDataLSTM(xTest, shifts), yTest.iloc[:-shifts + 1], prepareDataLSTM(xTrain, shifts), yTrain.iloc[
                                                                                                      :-shifts + 1]


def splitData(data, target, test_size=0.2):
    train_df = data.head(int(len(data) * (1 - test_size)))
    test_df = data.tail(int(len(data) * (test_size)))
    xTest = test_df.drop(target, axis=1)
    yTest = test_df[target]
    xTrain = train_df.drop(target, axis=1)
    yTrain = train_df[target]
    return xTest, yTest, xTrain, yTrain


def commonPreprocessing(data, enable_daytime_index=True, dailyCols=None,
                        monthlyCols=None, keepDailyAvg=None, keepMonthlyAvg=None):
    if enable_daytime_index:
        data = set_daytime_index(data)
    if monthlyCols is not None:
        data = monthly_diff(data, monthlyCols, keepMonthlyAvg)
    if dailyCols is not None:
        data = daily_diff(data, dailyCols, keepDailyAvg)
    return data


def preprocessingXGBoost(data, test_size, colums, shifts, negShifts, enable_daytime_index=True, dailyCols=None,
                         monthlyCols=None, keepDailyAvg=None, keepMonthlyAvg=None):
    data = commonPreprocessing(data, enable_daytime_index=enable_daytime_index, dailyCols=dailyCols,
                               monthlyCols=monthlyCols, keepDailyAvg=keepDailyAvg, keepMonthlyAvg=keepMonthlyAvg)
    data = slidingWindow(data, colums, shifts, negShifts)
    data = splitData(data, 'electricLoad', test_size)
    return data


def get_daylight_phase(row):
    city = LocationInfo("Berlin", "Germany", "Europe/Berlin", 52.5200, 13.4050)  # Bitte an Ihren Standort anpassen
    date_time = datetime(int(row['startDate_year']), int(row['startDate_month']), int(row['startDate_day']),
                         int(row['startDate_hour']), int(row['startDate_minute']))
    s = sun(city.observer, date=date_time.date())
    time = date_time.time()

    if s['dawn'].time() <= time < s['sunrise'].time():
        return 0
    elif s['sunrise'].time() <= time < s['noon'].time():
        return 1
    elif s['noon'].time() <= time < s['sunset'].time():
        return 2
    elif s['sunset'].time() <= time < s['dusk'].time():
        return 3
    else:
        return 4


def set_daytime_index(df):
    df['daylight_phase'] = df.apply(get_daylight_phase, axis=1)
    return df


def daily_diff(df, cols, keepAvg):
    df['date_str_day'] = df['startDate_year'].astype(str) + '-' + df['startDate_month'].astype(str).str.zfill(2) + '-' + \
                         df[
                             'startDate_day'].astype(str).str.zfill(2)
    i = 0
    for col in cols:
        daily_avg_temp = df.groupby('date_str_day')[col].mean().reset_index()
        daily_avg_temp.columns = ['date_str_day', 'avg_' + col + '_day']
        df = pd.merge(df, daily_avg_temp)
        # 96 = 4 per hour * 24 hours
        df[col + '_diff_prev_day'] = df['avg_' + col + '_day'] - df['avg_' + col + '_day'].shift(96)
        df.fillna(0, inplace=True)
        if not keepAvg[i]:
            df.drop('avg_' + col + '_day', axis=1, inplace=True)
        i += 1
    df.drop('date_str_day', axis=1, inplace=True)
    return df


def monthly_diff(df, cols, keepAvg):
    df['date_str_month'] = df['startDate_year'].astype(str) + '-' + df['startDate_month'].astype(str).str.zfill(2)
    i = 0
    for col in cols:
        daily_avg_temp = df.groupby('date_str_month')[col].mean().reset_index()
        daily_avg_temp.columns = ['date_str_month', 'avg_' + col + '_month']
        df = pd.merge(df, daily_avg_temp)
        # 4 per hour * 24 hours  * 30 days
        df[col + '_diff_prev_month'] = df['avg_' + col + '_month'] - df['avg_' + col + '_month'].shift(96 * 30)
        df.fillna(0, inplace=True)
        if not keepAvg[i]:
            df.drop('avg_' + col + '_month', axis=1, inplace=True)
        i += 1
    df.drop('date_str_month', axis=1, inplace=True)
    return df
