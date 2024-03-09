import pandas as pd


def read_load_curve_to_dataframe(
    file_path, sep=";", decimal=",", include_additional_columns=False
):
    """
    Reads a CSV file containing electric load data and converts it to a DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    sep : str, optional
        The separator used in the CSV file, by default ";".
    decimal : str, optional
        The decimal separator used in the CSV file, by default ",".
    include_additional_columns : bool, optional
        Whether to include additional columns (e.g., environmental data), by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the cleaned electric load data, with numerical features derived from the start date and electric load as a float.
    """
    column_names = ["startDate", "endDate", "electricLoad"]
    if include_additional_columns:
        column_names.extend(
            ["t1", "t2", "r1", "r2"]
        )  # Adjust based on actual additional columns needed

    df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=0, names=column_names)

    # Automatically detect date format
    for date_format in ("%d.%m.%Y %H:%M", "%d.%m.%y %H:%M"):
        try:
            df["startDate"] = pd.to_datetime(df["startDate"], format=date_format)
            break
        except ValueError:
            continue

    df = df.drop(columns="endDate").dropna()
    df = date_to_float(df, ["startDate"])
    df = clean_data(df)
    return df.dropna()


def get_data(dataset):
    """
    Fetches and processes data based on the specified dataset type.

    Parameters
    ----------
    dataset : str
        The dataset type to fetch and process.

    Returns
    -------
    pd.DataFrame
        The processed data merged with holiday information if applicable.
    """
    file_map = {
        "loadCurveOne": "../data/training_data_period_1.csv",
        "loadCurveTwo": "../data/training_data_period_2.csv",
        "loadCurveThree": "../data/training_data_period_3.csv",
        "loadCurveOneFull": "../data/training_data_period_1_full.csv",
        "loadCurveTwoFull": "../data/training_data_period_2_full.csv",
        "loadCurveThreeFull": "../data/training_data_period_3_full.csv",
    }

    if dataset not in file_map:
        raise ValueError("Dataset Not Supported")

    file_path = file_map[dataset]
    include_additional_columns = "Two" in dataset or "Three" in dataset
    load_curve = read_load_curve_to_dataframe(
        file_path, include_additional_columns=include_additional_columns
    )

    holiday = read_holiday_to_dataframe("../data/holiday.csv")
    return merge_holiday_and_load_curve(holiday, load_curve)


def read_holiday_to_dataframe(file_path, sep=";"):
    """
    Reads a CSV file containing holiday data and converts it to a DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing holiday data.
    sep : str, optional
        The separator used in the CSV file, by default ";".

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for holiday, school holiday indicator, and extracted date features (year, month, day).
    """
    df = pd.read_csv(
        file_path, sep=sep, header=0, names=["date", "holiday", "schoolHoliday"]
    )
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df["date_year"] = df["date"].dt.year
    df["date_month"] = df["date"].dt.month
    df["date_day"] = df["date"].dt.day
    df.drop("date", axis=1, inplace=True)
    return df.dropna()


def date_to_float(df, cols):
    """
    Converts datetime columns specified in 'cols' into multiple numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime columns to be converted.
    cols : list of str
        List of column names in 'df' that are datetime objects to be converted.

    Returns
    -------
    pd.DataFrame
        The DataFrame with original datetime columns replaced by numerical columns.
    """
    for col in cols:
        df["dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_minute"] = df[col].dt.minute
    df.drop(cols, axis=1, inplace=True)
    return df


def clean_data(df):
    """
    Cleans the provided DataFrame by replacing placeholder values for missing data with pandas NA.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with consistent numerical formatting and without NaN values.
    """
    df.replace("#NA!", pd.NA, inplace=True)
    df.replace({",": "."}, regex=True, inplace=True)
    df = df.dropna()
    df["electricLoad"] = df["electricLoad"].astype(float)
    df = df.dropna()
    return df


def merge_holiday_and_load_curve(holiday, loadCurveOne):
    """
    Merges two DataFrames: one containing holiday data and the other containing electric load data.

    Parameters
    ----------
    holiday : pd.DataFrame
        DataFrame containing holiday data with year, month, and day columns.
    loadCurveOne : pd.DataFrame
        DataFrame containing electric load data with corresponding year, month, and day columns.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing both electric load and holiday data on matching dates.
    """
    merged_df = pd.merge(
        left=loadCurveOne,
        right=holiday,
        left_on=["startDate_year", "startDate_month", "startDate_day"],
        right_on=["date_year", "date_month", "date_day"],
        how="inner",
    )
    merged_df.drop("date_year", axis=1, inplace=True)
    merged_df.drop("date_month", axis=1, inplace=True)
    merged_df.drop("date_day", axis=1, inplace=True)

    return merged_df
