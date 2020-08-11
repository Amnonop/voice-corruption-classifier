import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import librosa
from sklearn.utils import shuffle

from commons import SAMPLE_SIZE

CSV_FILENAME_PREFIX = 'siamese_speakers_{0}.csv'

DATASET_FILE_FORMAT = 'siamese_{0}_{1}.npy'

class SiameseLoader:
    _data_subsets = ['train', 'test']

    def __init__(self, root_dir: str, device = None):
        self._data = {}
        self._categories = {}
        self._classes_map = {}
        self._device = device

        for name in self._data_subsets:
            filepath = Path(root_dir).joinpath(DATASET_FILE_FORMAT.format(name, SAMPLE_SIZE))
            print(f'Loading data from {filepath}')
            x = np.load(filepath)
            self._data[name] = x

    def get_batch(self, batch_size: int, mode: str = 'train'):
        x = self._data[mode]
        num_classes = x.shape[0] 
        num_examples = x.shape[1]

        # Randomly select sample classes for the batch
        categories = np.random.choice(num_classes, size=(batch_size, ), replace=False)

        # Initialize the pairs tensor
        pairs = [np.zeros(shape=(batch_size, 1, SAMPLE_SIZE)) for i in range(2)]

        # Initialize targets so half of samples are from same class
        # 0 - SAME CLASS
        # 1 - DIFFERENT CLASS
        targets = np.ones(shape=(batch_size, ))
        targets[batch_size // 2:] = 0
        for i in range(batch_size):
            category = categories[i]
            first_index = np.random.randint(0, num_examples)
            pairs[0][i, :, :] = x[category, first_index].reshape(1, -1)

            second_index = np.random.randint(0, num_examples)
            if i >= batch_size // 2:
                second_category = category
            else:
                second_category = (category + np.random.randint(1, num_classes)) % num_classes

            pairs[1][i, :, :] = x[second_category, second_index].reshape(1, -1)

        pairs, targets = [torch.from_numpy(z).to(self._device) for z in [np.array(pairs).astype(np.float32), np.array(targets).astype(np.float32)]]

        return pairs, targets

    def make_oneshot_task(self, N, mode='test'):
        # 0 - SAME CLASS
        # 1 - DIFFERENT CLASS
        x = self._data[mode]
        num_classes = x.shape[0] 
        num_examples = x.shape[1]
        indices = np.random.randint(0, num_examples, size=(N, ))
        categories = np.random.choice(range(num_classes), size=(N, ), replace=False)

        true_category = categories[0]
        first_example, second_example = np.random.choice(num_examples, replace=False, size=(2, ))

        test_sample = np.asarray([x[true_category, first_example]] * N).reshape(N, 1, -1)

        support_set = x[categories, indices]
        support_set[0, :] = x[true_category, second_example]
        support_set = support_set.reshape(N, 1, -1)

        targets = np.ones(shape=(N, ))
        targets[0] = 0

        targets, test_sample, support_set = shuffle(targets, test_sample, support_set)
        pairs = [test_sample, support_set]

        pairs, targets = [torch.from_numpy(z).to(self._device) for z in [np.array(pairs).astype(np.float32), np.array(targets).astype(np.float32)]]

        return pairs, targets
