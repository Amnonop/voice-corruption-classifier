import torch
from pathlib import Path
import pandas
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import librosa
import numpy as np


AUDIO_LENGTH = 10000

class_ids = {'M': 0, 'F': 1}


class TrainTestSplitter:
    def __init__(self, csv_file, test_ratio):
        self.data_frame = pandas.read_csv(csv_file)
        self.train_set, self.test_set = train_test_split(self.data_frame, test_size=test_ratio, random_state=42)

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set


class AudioDataset(Dataset):
    def __init__(self, train_test_splitter, csv_file, root_dir, is_train=True):
        """
        Args:
            train_test_splitter (class): An instance of a TrainTestSplitter class
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio files
            is_train (bool): Indicates weather a training set is requested.
        """
        self.train_test_splitter = train_test_splitter
        self.is_train = is_train
        if self.is_train:
            self.audio_frame = train_test_splitter.get_train_set()
        else:
            self.audio_frame = train_test_splitter.get_test_set()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.audio_frame)

    def read_audio_file(self, filename):
        audio, sampling_rate = librosa.load(filename)
        #audio = audio.reshape(1, -1)

        # All samples should have the same size
        original_length = len(audio)
        if original_length < AUDIO_LENGTH:
            audio = np.concatenate((audio, np.zeros(shape=AUDIO_LENGTH - original_length)))
        elif original_length > AUDIO_LENGTH:
            audio = audio[0:AUDIO_LENGTH]
        return audio.reshape(1, -1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_filepath = Path(self.root_dir).joinpath(self.audio_frame.iloc[idx, 2])

        data = self.read_audio_file(audio_filepath)

        label = self.audio_frame.iloc[idx, 1]
        label_id = class_ids.get(label)

        sample = {'audio': data, 'label': label_id}

        return sample
