from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import librosa


class AudioDataset(Dataset):
    def __init__(self, x: DataFrame, y: DataFrame, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.x.iloc[idx]
        audio_path = Path(filename)
        signal, sampling_rate = librosa.load(audio_path)

        if self.transform:
            signal = self.transform(signal)

        label = self.y[idx]

        sample = {'signal': signal, 'label': label, 'filename': filename}

        return sample
