from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import librosa

from commons import *

class AudioDataset(Dataset):
    def __init__(self, data_frame: DataFrame, data_dir: str, transform=None):
        """
        Args:
            data_frame (pandas data frame): The data frame of the set.
            data_dir (string): Directory with all signals.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = data_frame
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data_frame.iloc[idx, 2]
        audio_path = Path(self.data_dir).joinpath(filename)
        signal, sampling_rate = librosa.load(audio_path)

        if self.transform:
            signal = self.transform(signal)

        label = self.data_frame.iloc[idx, 1]
        label_id = class_ids.get(label)

        sample = {'signal': signal, 'label': label_id, 'filename': filename}

        return sample
