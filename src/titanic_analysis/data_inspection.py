
"""
Functions for inspecting the Titanic datasets.

This module provides reusable functions for examining
missing values, frequent values, and unique values.
"""

import pandas as pd


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise missing values in each DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Table containing the number of missing values,
        percentage missing, and data type of each column.
    """
    total = df.isnull().sum()

    percent = (
        df.isnull().sum() / len(df) * 100
        if len(df) > 0
        else pd.Series(0.0, index=df.columns)
    )

    summary = pd.DataFrame({
        "Total": total,
        "Percent": percent,
        "Types": df.dtypes.astype(str),
    })

    return summary.T


def most_frequent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the most frequent value in each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Table containing the number of non-missing values,
        most frequent value, frequency, and percentage.
    """
    results = []

    for column in df.columns:
        counts = df[column].value_counts()

        total = df[column].count()

        if len(counts) > 0:
            most_frequent_item = counts.index[0]
            frequency = counts.iloc[0]
        else:
            most_frequent_item = None
            frequency = 0

        percentage = (
            frequency / total * 100
            if total > 0
            else 0.0
        )

        results.append({
            "Column": column,
            "Total": total,
            "Most frequent item": most_frequent_item,
            "Frequency": frequency,
            "Percent from total": percentage,
        })

    summary = pd.DataFrame(results)

    return summary.set_index("Column").T


def unique_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the unique non-missing values in each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Table containing the number of non-missing
        observations and the number of unique values.
    """
    summary = pd.DataFrame({
        "Total": df.count(),
        "Uniques": df.nunique(),
    })

    return summary.T
