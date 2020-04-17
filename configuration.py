import json
from pathlib import Path


class Configuration:
    def __init__(self, config_filename: str):
        config_json = load_config(config_filename)
        self.train_csv = config_json['train_csv']
        self.test_csv = config_json['test_csv']
        self._csv_filename = config_json['csv_filename']
        self._data_dir = config_json['data_dir']
        self._classify_by = config_json['classify_by']

    @property
    def csv_filename(self):
        return self._csv_filename

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def classify_by(self):
        return self._classify_by
    

def load_config(config_filename: str) -> Configuration:
    config_filepath = Path(config_filename)
    with open(config_filepath) as config_file:
        config = json.load(config_file)

    return config
