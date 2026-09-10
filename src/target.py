import pandas as pd


def count_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    для каждой даты считает таргет по формуле target_return = Close[next_day] / Open[next_day] - 1
    """
    df = df.sort_values("decision_day")
    df["target_return"] = df["Close"].shift(-1) / df["Open"].shift(-1) - 1
    return df.dropna(subset=["target_return"])
