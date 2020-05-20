import pickle
from pathlib import Path

import pandas as pd

import commons

CSV_FILENAME_PREFIX = 'siamese_speakers_{0}.csv'


class SiameseLoader:
    self._data_subsets = ['train', 'test']

    def __init__(self, csv_dir: str):
        self._data = {}
        self._categories = {}
        self._classes_map = {}

        for name in self._data_subsets:
            filepath = Path.cwd().joinpath(csv_dir).joinpath(DATA_FILENAME_PREFIX.format(name))
            print('Loading data from {filepath}')
            with open(filepath, 'rb') as file:
                (data, categories, classes_map) = pickle.load(file)
                self._data[name] = data
                self._categories[name] = categories
                self._classes_map[name] = classes_map
