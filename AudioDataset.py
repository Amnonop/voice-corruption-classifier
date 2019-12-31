import torch.utils.data as data
import os
from pathlib import Path
import pandas
from scipy.io import wavfile


class AudioDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path).joinpath('spkrinfo.csv')

        if not self._check_exists():
            raise RuntimeError(f'Dataset not found in {self.dataset_path}')

        self.csv_file = pandas.read_csv(self.dataset_path)#os.path.join(self.dataset_path, "spkrinfo.csv"))

    def _check_exists(self):
        return self.dataset_path.exists()#os.path.exists(os.path.join(self.dataset_path, "spkrinfo.csv"))

    def __len__(self):
        return self.csv_file.shape[0]

    def __getitem__(self, item):
        filename = self.csv_file['File'][item]

        rate, data = wavfile.read(self.dataset_path)

        label = self.csv_file['Sex'][item]

        return data, label
