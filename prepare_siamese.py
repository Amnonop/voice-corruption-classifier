import os
import shutil
import csv
import pickle
from pathlib import Path
import json

import numpy as np
from numpy import array
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import torch
from torch import Tensor

from commons import CONFIGS_DIR, DATA_FILENAME_PREFIX, SAMPLE_SIZE
from configuration import load_config
from dataset_transforms import TransformsComposer, ToTensor, Rescale

CONFIG_FILENAME = 'prepare_config.json'
CSV_FILENAME_PREFIX = 'siamese_speakers_{0}.csv'

class SpeakerData:
    def __init__(self, id: str, sex: str, dialect: str, files: list):
        self.id = id
        self.sex = sex
        self.dialect = dialect
        self.files = files


def get_directories(parent_dir: Path) -> list:
    return [child_entry for child_entry in parent_dir.iterdir() if child_entry.is_dir()]


def get_speakers_data(dataset_dir: str) -> dict:
    dataset_path = Path.cwd().joinpath(dataset_dir)
    speakers = {
        'train': [],
        'test': []
    }

    for mode_dir in get_directories(dataset_path):
        mode = mode_dir.name.lower()
        for dialect_dir in get_directories(mode_dir):
            for speaker_dir in get_directories(dialect_dir):
                speaker_sex = speaker_dir.name[0]
                speaker_id = speaker_dir.name[1:]
                speaker_files = [filename for filename in speaker_dir.iterdir() if filename.suffix == '.WAV']
                speaker = SpeakerData(speaker_id, speaker_sex, dialect_dir.name, speaker_files)
                speakers[mode].append(speaker)

    return speakers


def write_csv_file(csv_dir: str, csv_filename: str, speakers: list):
    rows = [["ID", "Sex", "Dialect", "File"]]
    for speaker in speakers:
        for speaker_file in speaker.files:
            rows.append([speaker.id, speaker.sex, speaker.dialect, speaker_file])

    csv_path = Path.cwd().joinpath(csv_dir).joinpath(csv_filename)

    print(f'Writing speakers data to {csv_filename}')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def write_files(csv_filename_prefix: str, csv_dir:str, speakers: dict):
    for mode in speakers.keys():
        csv_filename = csv_filename_prefix.format(mode)
        write_csv_file(csv_dir, csv_filename, speakers[mode])


def pickle_data(csv_filename_prefix: str, csv_dir:str):
    data = {'train': [], 'test': []}
    categories = {'train': [], 'test': []}
    classes_map = {'train': {}, 'test': {}}

    for name in data.keys():
        classes_map = {}

        csv_path = Path.cwd().joinpath(csv_dir).joinpath(csv_filename_prefix.format(name))
        print(f'Preparing to pickle {csv_path}')

        data_frame = pd.read_csv(csv_path, header=0)

        file_column = data_frame.columns.get_loc('File')
        category_column = data_frame.columns.get_loc('ID')

        x = data_frame.iloc[:, file_column].to_numpy()
        y = data_frame.iloc[:, category_column].to_numpy()

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        classes = encoder.classes_
        for i, category in enumerate(classes):
            classes_map[i] = category

        # Group by class
        categories = np.unique(y)
        data = [[] for i in range(len(categories))]
        for i, filename in enumerate(x):
            category = y[i]
            data[category].append(filename)

        data = array(data)

        pickle_filepath = Path.cwd().joinpath(csv_dir).joinpath(DATA_FILENAME_PREFIX.format(name))
        with open(pickle_filepath, 'wb') as pickle_file:
            print(f'Pickling {pickle_filepath}')
            pickle.dump((data, categories, classes_map), pickle_file)


def load_sample(sample_filename: str, transform: TransformsComposer = None) -> Tensor:
    sample_path = Path(sample_filename)
    signal, sampling_rate = librosa.load(sample_path)

    if transform:
        signal = transform(signal)

    return signal


def pickle_test_data(pickle_path: Path, target_path: Path, class_map_path: Path):
    transforms = TransformsComposer([Rescale(output_size=SAMPLE_SIZE)])

    with pickle_path.open(mode='rb') as pickle_file:
        print(f'Unpickling {pickle_path}')
        (data, categories, classes_map) = pickle.load(pickle_file)

    x = np.zeros(shape=(data.shape[0], 10, SAMPLE_SIZE))
    for i, speaker in enumerate(data):
        for j, filename in enumerate(speaker):
            signal = load_sample(filename, transforms)
            x[i, j, :] = signal

    with target_path.open(mode='wb') as target_file:
        print(f'Saving data to {target_path}')
        np.save(target_file, x)

    with class_map_path.open(mode="w") as class_map_file:
        print(f'Saving classes map to {class_map_path}')
        json.dump(classes_map, class_map_file)


def main():
    config_path = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = load_config(config_path)

    dataset_dir = config['dataset_dir']

    print(f'Getting speakers data from {dataset_dir}')
    # speakers = get_speakers_data(dataset_dir)

    csv_dir = config['csv_path']

    #write_files(CSV_FILENAME_PREFIX, csv_dir, speakers)

    # pickle_data(CSV_FILENAME_PREFIX, csv_dir)

    pickle_file = Path.cwd().joinpath(csv_dir).joinpath('siamese_train.pickle')
    target_file = Path.cwd().joinpath(csv_dir).joinpath(f'siamese_train_{SAMPLE_SIZE}.npy')
    class_map_file = Path.cwd().joinpath(csv_dir).joinpath('speakers_class_map_train.json')
    pickle_test_data(pickle_file, target_file, class_map_file)

    print('Done')


if __name__ == '__main__':
    main()





