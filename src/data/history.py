"""
Historical data loader — reads/writes collected game data from CSV.
"""

import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_game_data(filename: str = "game_log.csv") -> pd.DataFrame:
    """
    Load previously saved batter-game data from CSV.

    Raises:
        FileNotFoundError: If no saved data exists yet.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at {path}. Run: python -m src.data.collect_bulk"
        )
    return pd.read_csv(path)


def load_game_results(filename: str = "game_results.csv") -> pd.DataFrame:
    """Load game results data."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at {path}. Run: python -m src.data.collect_bulk"
        )
    return pd.read_csv(path)
