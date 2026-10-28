"""Load the student CSV and describe it.

Why a separate module? Loading files is where most beginner errors happen
(wrong path, empty file, wrong columns). Keeping that logic in one place lets
the rest of the program assume it receives a proper DataFrame.
"""

from pathlib import Path

import pandas as pd

import config


def load_students(path=config.RAW_CSV):
    """Read a student CSV into a DataFrame, with friendly error messages."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Tip: generate the dataset first "
            "(answer 'y' when the menu offers it, or run 'python src/data_generator.py')."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"'{path.name}' is empty - there is nothing to load.") from None
    except pd.errors.ParserError as problem:
        raise ValueError(f"'{path.name}' is not a valid CSV file: {problem}") from None

    missing_columns = [c for c in config.COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"'{path.name}' is missing required column(s): {', '.join(missing_columns)}"
        )

    return df[config.COLUMNS].copy()


def overview(df):
    """Return basic facts about a DataFrame as a dictionary."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


def dtypes_table(df):
    """A tidy table: column, data type, number of missing values."""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing": df.isnull().sum().to_numpy(),
    })


def summary_statistics(df):
    """Summary statistics (count, mean, std, min, quartiles, max) for numbers."""
    return df.describe().T.reset_index().rename(columns={"index": "column"})


def series_vs_dataframe_demo(df):
    """Return one Series and one DataFrame taken from `df` for comparison.

    * A Series is ONE column: a labelled 1-dimensional array.
    * A DataFrame is a TABLE: several Series sharing the same row index.
    """
    one_column = df["final_marks"]            # single brackets -> Series
    two_columns = df[["name", "final_marks"]]  # double brackets -> DataFrame
    return one_column, two_columns
