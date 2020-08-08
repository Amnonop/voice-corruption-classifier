from pathlib import Path

import numpy as np

import torch

from commons import SAMPLE_SIZE

DATASET_FILENAME = 'siamese_{0}_{SAMPLE_SIZE}.npy'

CLASS_SAMPLES = 10

class AudioNShot:
    def __init__(self, root: str, batch_size: int, n_way: int, k_shot: int, k_query: int, sample_size: int, device=None):
        self.resize = sample_size
        self.device = device
        data = ['train', 'test']
        self.x_train = np.load(Path(root).joinpath(DATASET_FILENAME.format('train')))
        self.x_test = np.load(Path(root).joinpath(DATASET_FILENAME.format('train')))

        self.batch_size = batch_size
        self.num_classes = self.x_train.shape[0] + self.x_test.shape[0]
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        assert(k_shot + k_query) <= 20

        # Save pointer to currently read batch
        self.indexes = { 'train': 0, 'test': 0 }

        # Cache the original data
        self.datasets = { 'train': self.x_train, 'test': self.x_test }
        print(f'DB: train {self.x_train.shape} test {self.x_test.shape}')

        # Create a cache for the current epoch
        self.datasets_cache = {
            'train': self.load_data_cache(self.datasets['train']),
            'test': self.load_data_cache(self.datasets['test'])
        }

    def load_data_cache(self, data_pack):
        set_size = self.k_shot * self.n_way
        query_size = self.k_query * self.n_way
        data_cache = []

        for sample in range(10):
            x_support_sets, y_support_sets = []
            x_query_sets, y_query_sets = []

            for i in range(self.batch_size):
                x_support, y_support = []
                x_query, y_query = []
                selected_classes = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, current_class in enumerate(selected_classes):
                    selected_audios = np.random.choice(CLASS_SAMPLES, self.k_shot + self.k_query, False)

                    x_support.append(data_pack[current_class][selected_audios[:self.k_shot]])
                    x_query.append(data_pack[current_class][selected_audios[self.k_shot:]])

                    y_support.append(j * self.k_shot)
                    y_query.append(j * self.k_query)

                # Shuffle this batch
                permutations = np.random.permutation(self.n_way * self.k_shot)
                x_support = np.array(x_support).reshape(self.n_way * self.k_shot, 1, self.resize)[permutations]
                y_support = np.array(y_support).reshape(self.n_way * self.k_shot)[permutations]

                permutations = np.random.permutation(self.n_way * self.k_query)
                x_query = np.array(x_query).reshape(self.n_way * self.k_query, 1, self.resize)[permutations]
                y_query = np.array(y_query).reshape(self.n_way * self.k_query)[permutations]

                x_support_sets.append(x_support)
                y_support_sets.append(y_support)
                x_query_sets.append(x_query)
                y_query_sets.append(y_query)

            x_support_sets = np.array(x_support_sets).astype(np.float32).reshape(self.batch_size, set_size, 1, self.resize)
            y_support_sets = np.array(y_support_sets).astype(np.int).reshape(self.batch_size, set_size)

            x_query_sets = np.array(x_query_sets).astype(np.float32).reshape(self.batch_size, query_size, 1, self.resize)
            y_query_sets = np.array(y_query_sets).astype(np.int).reshape(self.batch_size, query_size)

            x_support_sets, y_support_sets, x_query_sets, y_query_sets = [
                torch.from_numpy(z).to(self.device) for z in [x_support_sets, y_support_sets, x_query_sets, y_query_sets]
                ]

            data_cache.append([x_support_sets, y_support_sets, x_query_sets, y_query_sets])

        return data_cache

    def next(self, mode='train'):
        # Check if cache update is needed
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
            