import torch.utils.data as data
import os
import pandas
from scipy.io import wavfile


class AudioDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        if not self._check_exists():
            raise RuntimeError(f'Dataset not found in {self.dataset_path}')

        self.csv_file = pandas.read_csv(os.path.join(self.dataset_path, "spkrinfo.csv"))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.dataset_path, "spkrinfo.csv"))

    def __len__(self):
        return self.csv_file.shape[0]

    def __getitem__(self, item):
        filename = self.csv_file["File"][item]

        rate, data = wavfile.read(os.path.join(self.dataset_path, filename))

        label = self.csv_file["ID"][item]

        return data, label
