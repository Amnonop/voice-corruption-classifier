import random
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import librosa

from sample_logger import SampleLogger, DatasetItem


class AudioDataset(Dataset):
    def __init__(self, x: DataFrame, y: DataFrame, classes: dict, sample_logger: SampleLogger, transform=None):
        self.x = x
        self.y = y
        self.classes = classes
        self.dataset_by_class = self.group_dataset(x, y, classes)
        self.transform = transform
        self._sample_logger = sample_logger

    def group_dataset(self, x: DataFrame, y: DataFrame, classes: dict) -> dict:
        dataset_by_class = {}
        for class_id in classes.keys():
            dataset_by_class[class_id] = []

        for i, class_id in enumerate(y):
            dataset_by_class[class_id].append(x[i])
        return dataset_by_class

    def __len__(self):
        return 8
        # return len(self.x)

    def get_second_class(self, class_index: int) -> int:
        second_class_index = random.randint(0, len(self.classes) - 1)
        while class_index == second_class_index:
            second_class_index = random.randint(0, len(self.classes) - 1)
        return second_class_index
    
    def load_random_sample(self, class_id: int) -> dict:
        filename = random.choice(self.dataset_by_class[class_id])
        sample_path = Path(filename)
        signal, sampling_rate = librosa.load(sample_path)

        if self.transform:
            signal = self.transform(signal)

        return {
            'signal': signal,
            'filename': filename,
            'label': self.classes[class_id]
            }

    def get_sample_pair(self, same_class: bool) -> Tuple[dict, dict]:
        class_id = random.randint(0, len(self.classes) - 1)
        first_sample = self.load_random_sample(class_id)

        if same_class:
            second_sample = self.load_random_sample(class_id)
        else:
            second_class_id = self.get_second_class(class_id)
            second_sample = self.load_random_sample(second_class_id)

        self._log_sample_pair(first_sample, second_sample, same_class)

        return first_sample, second_sample

    def _log_sample_pair(self, first_sample: dict, second_sample: dict, same_label:bool):
        first = DatasetItem(first_sample['filename'], first_sample['label'])
        second = DatasetItem(second_sample['filename'], second_sample['label'])
        self._sample_logger.add_sample_pair(first, second, same_label)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if index % 2 == 1:
            label = 1
            first_sample, second_sample = self.get_sample_pair(same_class=True)
        else:
            label = 0
            first_sample, second_sample = self.get_sample_pair(same_class=False)

        return first_sample, second_sample, torch.from_numpy(np.array([label], dtype=np.float32))
