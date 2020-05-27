import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import librosa
from sklearn.utils import shuffle

from commons import DATA_FILENAME_PREFIX, SAMPLE_SIZE
from dataset_transforms import TransformsComposer

CSV_FILENAME_PREFIX = 'siamese_speakers_{0}.csv'


class SiameseLoader:
    _data_subsets = ['train', 'test']

    def __init__(self, csv_dir: str, transform:TransformsComposer = None):
        self._data = {}
        self._categories = {}
        self._classes_map = {}
        self._transform = transform

        for name in self._data_subsets:
            filepath = Path.cwd().joinpath(csv_dir).joinpath(DATA_FILENAME_PREFIX.format(name))
            print(f'Loading data from {filepath}')
            with open(filepath, 'rb') as file:
                (data, categories, classes_map) = pickle.load(file)
                self._data[name] = data
                self._categories[name] = categories
                self._classes_map[name] = classes_map

    def _load_sample(self, sample_filename: str) -> Tensor:
        sample_path = Path(sample_filename)
        signal, sampling_rate = librosa.load(sample_path)

        if self._transform:
            signal = self._transform(signal)

        return signal

    def get_batch(self, batch_size: int, mode: str = 'train'):
        x = self._data[mode]
        num_classes, num_examples = x.shape

        # Randomly select sample classes for the batch
        categories = np.random.choice(num_classes, size=(batch_size, ), replace=False)

        # Initialize the pairs tensor
        pairs = [torch.zeros((batch_size, 1, SAMPLE_SIZE)) for i in range(2)]

        # Initialize targets so half of samples are from same class
        # 0 - SAME CLASS
        # 1 - DIFFERENT CLASS
        targets = torch.ones((batch_size, ))
        targets[batch_size // 2:] = 0
        for i in range(batch_size):
            category = categories[i]
            first_index = np.random.randint(0, num_examples)
            pairs[0][i, :, :] = self._load_sample(x[category, first_index])

            second_index = np.random.randint(0, num_examples)
            if i >= batch_size // 2:
                second_category = category
            else:
                second_category = (category + np.random.randint(1, num_classes)) % num_classes

            pairs[1][i, :, :] = self._load_sample(x[second_category, second_index])

        return pairs, targets

    def make_oneshot_task(self, N, mode='test'):
        # 0 - SAME CLASS
        # 1 - DIFFERENT CLASS
        x = self._data[mode]
        num_classes, num_examples = x.shape
        indices = np.random.randint(0, num_examples, size=(N, ))
        categories = np.random.choice(range(num_classes), size=(N, ), replace=False)

        true_category = categories[0]
        first_example, second_example = np.random.choice(num_examples, replace=False, size=(2, ))

        test_sample = torch.stack([self._load_sample(x[true_category, first_example])] * N)

        support_set = torch.stack([self._load_sample(sample_filename) for sample_filename in x[categories, indices]])
        support_set[0, :, :] = self._load_sample(x[true_category, second_example])

        targets = torch.ones((N, ))
        targets[0] = 0

        targets, test_sample, support_set = shuffle(targets, test_sample, support_set)
        pairs = [test_sample, support_set]

        return pairs, targets
