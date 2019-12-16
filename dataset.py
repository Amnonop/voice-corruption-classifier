import os
from os import listdir
import numpy as np
from sklearn import model_selection
from scipy.io import wavfile

DATASET_PATH = "data/"


class Dataset:
    audio_files = []
    train_set = []
    test_set = []

    def __init__(self):
        self.load_audio_filenames()
        self.train_test_split()

    def load_audio_filenames(self):
        audio_files = [os.path.join(DATASET_PATH, filename) for filename in listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, filename))]

    def train_test_split(self):
        train_set_filenames, test_set_filenames = model_selection.train_test_split(self.audio_files, test_size=0.2, random_state=42)

        self.train_set = self.read_audio_files(train_set_filenames)
        self.test_set = self.read_audio_files(test_set_filenames)

    def read_audio_files(self, filenames):
        return [wavfile.read(filename) for filename in filenames]






