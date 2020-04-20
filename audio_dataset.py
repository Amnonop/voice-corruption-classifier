import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import librosa


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

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if index % 2 == 1:
            label = 1.0
            class_index = random.randint(0, len(self.classes) - 1)
            file1 = random.choice(self.dataset_by_class[class_index])
            file2 = randome.choice(self.dataset_by_class[class_index])
        else:
            label = 0.0
            class_index1 = random.randint(0, len(self.classes) - 1)
            class_index2 = random.randint(0, len(self.classes) - 1)
            while class_index1 == class_index2:
                class_index2 = random.randint(0, len(self.classes) - 1)
            file1 = randome.choice(self.dataset_by_class[class_index1])
            file2 = randome.choice(self.dataset_by_class[class_index2])
        audio_path = Path(filename)
        signal, sampling_rate = librosa.load(audio_path)

        if self.transform:
            signal = self.transform(signal)

        label = self.y[idx]

        sample = {'signal': signal, 'label': label, 'filename': filename}

        return sample
