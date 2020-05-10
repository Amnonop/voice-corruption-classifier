import random
from typing import Tuple
from pathlib import Path

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import librosa

from sample_logger import SampleLogger, DatasetItem


class AudioDataset(Dataset):
    def __init__(self, x: DataFrame, y: ndarray, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filename = self.x.iloc[index]
        label = self.y[index]

        sample_path = Path(filename)
        signal, sampling_rate = librosa.load(sample_path)

        if self.transform:
            signal = self.transform(signal)

        return {
            'filename': filename,
            'signal': signal,
            'label': label
        }
