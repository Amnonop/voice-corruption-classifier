from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from commons import CONFIGS_DIR
from configuration import load_config

CONFIG_FILENAME = 'prepare_config.json'
CSV_FILENAME = 'speakers.csv'


def print_statistics(data: DataFrame, class_column: int) -> None:
    encoder = LabelEncoder()
    classes_column = data.iloc[:, class_column]
    classes_encoded = encoder.fit_transform(classes_column)
    classes_count = np.bincount(classes_encoded)

    print(f'Total entries: {len(classes_column)}')
    print('CLASS    COUNT       %')
    print('============================')
    for i, class_count in enumerate(classes_count):
        class_name = encoder.classes_[i]
        class_percent = (class_count / len(classes_column)) * 100
        print('{:s}       {:d}       {:.2f}%'.format(class_name, class_count, class_percent))


def split(data: DataFrame, split_category_column: int, test_ratio: float = 0.2) -> Tuple[ndarray, ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in splitter.split(data, data.iloc[:, split_category_column]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]
    return train_set, test_set

def main():
    config_path = Path(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = load_config(config_path)

    csv_dir = config['csv_path']
    csv_path = Path(csv_dir).joinpath(CSV_FILENAME)

    print(f'Reading data from {csv_path}')
    data = pd.read_csv(csv_path, header=0)

    split_category = config['split_by']
    test_ratio = config['test_ratio']

    print(f'Splitting by {split_category} with test ratio of {test_ratio}')
    split_column = data.columns.get_loc(split_category)

    print_statistics(data, split_column)

    train_set, test_set = split(data, split_column, test_ratio)

    print()
    print('TRAIN')
    print('=====')
    print_statistics(train_set, split_column)

    print()
    print('TEST')
    print('=====')
    print_statistics(test_set, split_column)

    train_csv_filename = f'train_by_{split_category.lower()}.csv'
    csv_path = Path(csv_dir).joinpath(train_csv_filename)

    print()
    print(f'Writing training set to {csv_path}')
    pd.DataFrame(train_set).to_csv(csv_path)

    test_csv_filename = f'test_by_{split_category.lower()}.csv'
    csv_path = Path(csv_dir).joinpath(test_csv_filename)

    print(f'Writing testing set to {csv_path}')
    pd.DataFrame(train_set).to_csv(csv_path)


if __name__ == '__main__':
    main()