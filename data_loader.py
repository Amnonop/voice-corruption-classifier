from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from configuration import Configuration


class DataLoader:
    def __init__(self, config: Configuration):
        self._train_csv_filename = config['train_csv']
        self._test_csv_filename = config['test_csv']
        self._category = config.classify_by

    def get_train_set(self) -> Tuple[DataFrame, DataFrame]:
        return self._load_data(self._train_csv_filename)

    def get_test_set(self) -> Tuple[DataFrame, DataFrame]:
        return self._load_data(self._test_csv_filename)

    def _load_data(self, csv_filename: str) -> Tuple[DataFrame, DataFrame]:
        csv_path = Path(csv_filename)
        data_frame = pd.read_csv(csv_path, header=0)

        filename_column = data_frame.columns.get_loc('Filename')
        category_column = data_frame.columns.get_loc(self._category)

        x = data_frame.iloc[:, filename_column]
        y = data_frame.iloc[:, category_column]

        return x, y
