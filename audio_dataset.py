import random
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import librosa


class AudioSample:
    def __init__(self, signal: array, filename: str, label: str):
        self.signal = signal
        self.filename = filename
        self.label = label


class AudioDataset(Dataset):
    def __init__(self, x: DataFrame, y: DataFrame, classes: dict, transform=None):
        self.x = x
        self.y = y
        self.classes = classes
        self.dataset_by_class = self.group_dataset(x, y, classes)
        self.transform = transform

    def group_dataset(self, x: DataFrame, y: DataFrame, classes: dict) -> dict:
        dataset_by_class = {}
        for class_id in classes.keys():
            dataset_by_class[class_id] = []

        for i, class_id in enumerate(y):
            dataset_by_class[class_id].append(x[i])
        return dataset_by_class

    def __len__(self):
        return len(self.x)

    def get_second_class(self, class_index: int) -> int:
        second_class_index = random.randint(0, len(self.classes) - 1)
        while class_index == second_class_index:
            second_class_index = random.randint(0, len(self.classes) - 1)
        return second_class_index
    
    def load_random_sample(self, class_id: int) -> AudioSample:
        filename = random.choice(self.dataset_by_class[class_id])
        sample_path = Path(filename)
        signal, sampling_rate = librosa.load(sample_path)

        if self.transform:
            signal = self.transform(signal)

        return AudioSample(signal, filename, self.classes[class_id])

    def get_sample_pair(self, same_class: bool) -> Tuple[AudioSample, AudioSample]:
        class_id = random.randint(0, len(self.classes) - 1)
        first_sample = self.load_random_sample(class_id)

        if same_class:
            second_sample = self.load_random_sample(class_id)
        else:
            second_class_id = self.get_second_class(class_id)
            second_sample = self.load_random_sample(second_class_id)

        return first_sample, second_sample

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if index % 2 == 1:
            label = 1.0
            first_sample, second_sample = self.get_sample_pair(same_class=True)
        else:
            label = 0.0
            first_sample, second_sample = self.get_sample_pair(same_class=False)

        return first_sample, second_sample, torch.from_numpy(np.array([label], dtype=np.float32))
