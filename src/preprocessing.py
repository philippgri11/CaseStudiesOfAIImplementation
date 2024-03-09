import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from astral.sun import sun
from astral import LocationInfo


def sliding_window(df, columns, shifts, neg_shifts):
    """
        Applies a sliding window transformation to specified columns in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        columns : list of str
            Columns to which the sliding window transformation is applied.
        shifts : int
            Number of positive shifts (forward in time) to apply for the window.
        neg_shifts : int
            Number of negative shifts (backward in time) to apply for the window.

        Returns
        -------
        pd.DataFrame
            The DataFrame with additional shifted columns for the specified window.
    """
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
    """
        Prepares data for LSTM models by creating sequences from DataFrame rows.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        shifts : int
            Number of rows to include in each sequence.

        Returns
        -------
        np.ndarray
            A 3D array suitable for LSTM input where each sequence is a time step.
    """
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
    """
    Preprocesses data specifically for LSTM models, including scaling and sequence creation.

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    shifts : int
        The number of time steps for each sequence.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    val_size : float, optional
        Proportion of the dataset to include in the validation split.
    target : str, optional
        The target column in the dataset.
    enable_daytime_index : bool, optional
        If True, calculates and includes the daytime index.
    daily_cols : list of str, optional
        Columns for which daily differences are calculated.
    monthly_cols : list of str, optional
        Columns for which monthly differences are calculated.
    keep_daily_avg : list of bool, optional
        Specifies whether to keep the daily averages in the dataset.
    keep_monthly_avg : list of bool, optional
        Specifies whether to keep the monthly averages in the dataset.

    Returns
    -------
    tuple
        A tuple containing the processed data splits for training, validation, and testing, each as a numpy array.
    """
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
    x_val_scaler = StandardScaler()
    x_test = pd.DataFrame(x_test_scaler.fit_transform(x_test.to_numpy()))
    x_train = pd.DataFrame(x_train_scaler.fit_transform(x_train.to_numpy()))
    x_val = pd.DataFrame(x_val_scaler.fit_transform(x_val.to_numpy()))
    return (
        prepare_data_lstm(x_train, shifts),
        y_train.iloc[: -shifts + 1],
        prepare_data_lstm(x_val, shifts),
        y_val.iloc[: -shifts + 1],
        prepare_data_lstm(x_test, shifts),
        y_test.iloc[: -shifts + 1],
    )


def split_data(data, target, test_size, val_size):
    """
        Splits the data into training, validation, and testing sets.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        target : str
            The target column in the dataset.
        test_size : float
            Proportion of the dataset to include in the test split.
        val_size : float
            Proportion of the dataset to include in the validation split.

        Returns
        -------
        tuple
            A tuple containing the feature and target splits for training, validation, and testing.
    """
    train_size = 1 - (test_size + val_size)

    if train_size < 0:
        raise ValueError(
            "Die Summe von test_size und val_size muss kleiner als 1 sein."
        )

    train_end = int(len(data) * train_size)
    val_end = train_end + int(len(data) * val_size)

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

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
    """
        Applies common preprocessing steps including daytime index calculation and difference calculations.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        enable_daytime_index : bool, optional
            If True, calculates and includes the daytime index.
        daily_cols : list of str, optional
            Columns for which daily differences are calculated.
        monthly_cols : list of str, optional
            Columns for which monthly differences are calculated.
        keep_daily_avg : list of bool, optional
            Specifies whether to keep the daily averages in the dataset.
        keep_monthly_avg : list of bool, optional
            Specifies whether to keep the monthly averages in the dataset.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
    """
    if enable_daytime_index:
        data = set_daytime_index(data)
    if monthly_cols is not None:
        data = monthly_diff(data, monthly_cols, keep_monthly_avg)
    if daily_cols is not None:
        data = daily_diff(data, daily_cols, keep_daily_avg)
    return data


def add_rolling_average_electric_load(df, column, window_size):
    """
        Adds a rolling average column to the DataFrame based on specified window size.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        column : str
            The column on which the rolling average is calculated.
        window_size : int
            The size of the rolling window.

        Returns
        -------
        pd.DataFrame
            The DataFrame with an added column for the rolling average.
    """
    df[f"{column}_rolling_avg_{window_size}"] = (
        df[column].shift(1).rolling(window=window_size).mean()
    )
    return df


def preprocessing(
    data,
    test_size,
    val_size,
    columns,
    shifts,
    neg_shifts,
    enable_daytime_index=True,
    daily_cols=None,
    monthly_cols=None,
    keep_daily_avg=None,
    keep_monthly_avg=None,
    load_lag=0,
):
    """
        Preprocesses data specifically for XGBoost models, including scaling and feature engineering.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        test_size : float
            Proportion of the dataset to include in the test split.
        val_size : float
            Proportion of the dataset to include in the validation split.
        columns : list of str
            Columns to be included in the sliding window transformation.
        shifts : int
            Number of positive shifts for the sliding window.
        neg_shifts : int
            Number of negative shifts for the sliding window.
        enable_daytime_index : bool, optional
            If True, calculates and includes the daytime index.
        daily_cols : list of str, optional
            Columns for which daily differences are calculated.
        monthly_cols : list of str, optional
            Columns for which monthly differences are calculated.
        keep_daily_avg : list of bool, optional
            Specifies whether to keep the daily averages in the dataset.
        keep_monthly_avg : list of bool, optional
            Specifies whether to keep the monthly averages in the dataset.
        load_lag : int, optional
            The lag size for calculating the rolling average of the electric load.

        Returns
        -------
        tuple
            A tuple containing the feature and target splits for training, validation, and testing.
    """
    data = common_preprocessing(
        data,
        enable_daytime_index=enable_daytime_index,
        daily_cols=daily_cols,
        monthly_cols=monthly_cols,
        keep_daily_avg=keep_daily_avg,
        keep_monthly_avg=keep_monthly_avg,
    )
    data = sliding_window(data, columns, shifts, neg_shifts)
    print(f"load_lag= {load_lag}")
    data = add_rolling_average_electric_load(data, "electricLoad", load_lag)
    data = data.dropna()
    data = split_data(data, "electricLoad", test_size, val_size)
    return data


def get_daylight_phase(row):
    """
        Calculates the daylight phase for a given row based on the sunrise and sunset times.

        Parameters
        ----------
        row : pd.Series
            A row of the DataFrame, expected to contain date and time information.

        Returns
        -------
        int
            An integer representing the daylight phase.
    """
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
    """
        Adds a column to the DataFrame representing the daylight phase for each row.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with an added 'daylight_phase' column.
    """
    df["daylight_phase"] = df.apply(get_daylight_phase, axis=1)
    return df


def daily_diff(df, cols, keep_avg):
    """
        Calculates the daily difference for specified columns and adds them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        cols : list of str
            Columns for which the daily difference is calculated.
        keep_avg : list of bool
            Specifies whether to keep the daily averages in the dataset.

        Returns
        -------
        pd.DataFrame
            The DataFrame with added columns for daily differences.
    """
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
    """
        Calculates the monthly difference for specified columns and adds them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        cols : list of str
            Columns for which the monthly difference is calculated.
        keepAvg : list of bool
            Specifies whether to keep the monthly averages in the dataset.

        Returns
        -------
        pd.DataFrame
            The DataFrame with added columns for monthly differences.
    """
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
