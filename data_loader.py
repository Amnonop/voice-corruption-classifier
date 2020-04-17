from pathlib import Path
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit


class DataLoader:
    def __init__(self, csv_filename: str):
        self._csv_filename = csv_filename
        self._data_frame = self.load_data()

    def load_data(self) -> DataFrame:
        csv_path = Path(self._csv_filename)
        data_frame = pd.read_csv(csv_path, header=0)
        return data_frame

    def split(self, test_ratio=0.2):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in splitter.split(self.data, self.data.iloc[:, 1]):
            train_set = self.data.loc[train_index]
            test_set = self.data.loc[test_index]
        return train_set, test_set

    @property
    def data(self):
        return self._data_frame
