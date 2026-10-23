
"""
Functions for loading the Titanic datasets.
"""

from pathlib import Path

import pandas as pd


def load_titanic_data(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Titanic training and test datasets.

    Parameters
    ----------
    train_path : str or Path
        Path to the training CSV file.
    test_path : str or Path
        Path to the test CSV file.

    Returns
    -------
    tuple of pandas.DataFrame
        Training and test datasets.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
