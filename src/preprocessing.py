import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from astral.sun import sun
from astral import LocationInfo


def sliding_window(df, columns, shifts, neg_shifts):
    if shifts > 0:
        for shift in range(1, shifts + 1):
            for col in np.asarray(columns).astype("str"):
                df[col + str(shift)] = df[col].shift(shift)
    if neg_shifts < 0:
        for negShift in range(neg_shifts, 0):
            for col in columns:
                df[col + str(negShift)] = df[col].shift(negShift)
    df = df.dropna()
    return df


def prepare_data_lstm(df, shifts):
    data = []
    for i in range(len(df) - shifts + 1):
        seq = df.iloc[i: i + shifts].values
        data.append(seq)
    return np.array(data)


def preprocessing_lstm(
    data,
    shifts,
    test_size=0.2,
    val_size=0.1,
    target="electricLoad",
    enable_daytime_index=True,
    daily_cols=None,
    monthly_cols=None,
    keep_daily_avg=None,
    keep_monthly_avg=None,
):
    data = common_preprocessing(
        data,
        enable_daytime_index=enable_daytime_index,
        daily_cols=daily_cols,
        monthly_cols=monthly_cols,
        keep_daily_avg=keep_daily_avg,
        keep_monthly_avg=keep_monthly_avg,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        data, target, test_size=test_size, val_size=val_size
    )
    x_test_scaler = StandardScaler()
    x_train_scaler = StandardScaler()
    # todo one norm
    x_test = pd.DataFrame(x_test_scaler.fit_transform(x_test.to_numpy()))
    x_train = pd.DataFrame(x_train_scaler.fit_transform(x_train.to_numpy()))
    return (
        prepare_data_lstm(x_test, shifts),
        y_test.iloc[: -shifts + 1],
        prepare_data_lstm(x_train, shifts),
        y_train.iloc[: -shifts + 1],
    )


def split_data(data, target, test_size, val_size):
    # Berechne die Größe für Trainingsdatensatz
    train_size = 1 - (test_size + val_size)

    # Stelle sicher, dass die Größenangaben gültig sind
    if train_size < 0:
        raise ValueError(
            "Die Summe von test_size und val_size muss kleiner als 1 sein."
        )

    # Splitte die Daten in Trainings-, Validierungs- und Testdatensätze
    train_end = int(len(data) * train_size)
    val_end = train_end + int(len(data) * val_size)

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    # Bereite die Trainings-, Validierungs- und Testdatensätze vor
    x_train = train_df.drop(target, axis=1)
    y_train = train_df[target]

    x_val = val_df.drop(target, axis=1)
    y_val = val_df[target]

    x_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    return x_train, y_train, x_val, y_val, x_test, y_test


def common_preprocessing(
    data,
    enable_daytime_index=True,
    daily_cols=None,
    monthly_cols=None,
    keep_daily_avg=None,
    keep_monthly_avg=None,
):
    if enable_daytime_index:
        data = set_daytime_index(data)
    if monthly_cols is not None:
        data = monthly_diff(data, monthly_cols, keep_monthly_avg)
    if daily_cols is not None:
        data = daily_diff(data, daily_cols, keep_daily_avg)
    return data


def add_rolling_average_electric_load(df, column, window_size):
    df[f"{column}_rolling_avg_{window_size}"] = (
        df[column].shift(1).rolling(window=window_size).mean()
    )
    return df


def preprocessing_xg_boost(
    data,
    test_size,
    val_size,
    columns,
    shifts,
    negShifts,
    enable_daytime_index=True,
    daily_cols=None,
    monthly_cols=None,
    keep_daily_avg=None,
    keep_monthly_avg=None,
    load_lag=0,
):
    data = common_preprocessing(
        data,
        enable_daytime_index=enable_daytime_index,
        daily_cols=daily_cols,
        monthly_cols=monthly_cols,
        keep_daily_avg=keep_daily_avg,
        keep_monthly_avg=keep_monthly_avg,
    )
    data = sliding_window(data, columns, shifts, negShifts)
    print(f"loadLag= {load_lag}")
    data = add_rolling_average_electric_load(data, "electricLoad", load_lag)
    data = split_data(data, "electricLoad", test_size, val_size)
    return data


def get_daylight_phase(row):
    city = LocationInfo("Berlin", "Germany", "Europe/Berlin", 52.5200, 13.4050)
    date_time = datetime(
        int(row["startDate_year"]),
        int(row["startDate_month"]),
        int(row["startDate_day"]),
        int(row["startDate_hour"]),
        int(row["startDate_minute"]),
    )
    s = sun(city.observer, date=date_time.date())
    time = date_time.time()

    if s["dawn"].time() <= time < s["sunrise"].time():
        return 0
    elif s["sunrise"].time() <= time < s["noon"].time():
        return 1
    elif s["noon"].time() <= time < s["sunset"].time():
        return 2
    elif s["sunset"].time() <= time < s["dusk"].time():
        return 3
    else:
        return 4


def set_daytime_index(df):
    df["daylight_phase"] = df.apply(get_daylight_phase, axis=1)
    return df


def daily_diff(df, cols, keep_avg):
    df["date_str_day"] = (
        df["startDate_year"].astype(str)
        + "-"
        + df["startDate_month"].astype(str).str.zfill(2)
        + "-"
        + df["startDate_day"].astype(str).str.zfill(2)
    )
    i = 0
    for col in cols:
        daily_avg_temp = df.groupby("date_str_day")[col].mean().reset_index()
        daily_avg_temp.columns = ["date_str_day", "avg_" + col + "_day"]
        df = pd.merge(df, daily_avg_temp)
        # 96 = 4 per hour * 24 hours
        df[col + "_diff_prev_day"] = df["avg_" + col + "_day"] - df[
            "avg_" + col + "_day"
        ].shift(96)
        df.fillna(0, inplace=True)
        if not keep_avg[i]:
            df.drop("avg_" + col + "_day", axis=1, inplace=True)
        i += 1
    df.drop("date_str_day", axis=1, inplace=True)
    return df


def monthly_diff(df, cols, keepAvg):
    df["date_str_month"] = (
        df["startDate_year"].astype(str)
        + "-"
        + df["startDate_month"].astype(str).str.zfill(2)
    )
    i = 0
    for col in cols:
        daily_avg_temp = df.groupby("date_str_month")[col].mean().reset_index()
        daily_avg_temp.columns = ["date_str_month", "avg_" + col + "_month"]
        df = pd.merge(df, daily_avg_temp)
        # 4 per hour * 24 hours  * 30 days
        df[col + "_diff_prev_month"] = df["avg_" + col + "_month"] - df[
            "avg_" + col + "_month"
        ].shift(96 * 30)
        df.fillna(0, inplace=True)
        if not keepAvg[i]:
            df.drop("avg_" + col + "_month", axis=1, inplace=True)
        i += 1
    df.drop("date_str_month", axis=1, inplace=True)
    return df
