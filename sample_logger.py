import json
from json import JSONEncoder
from typing import Dict
from pathlib import Path


class DatasetItem:
    def __init__(self, filename: str, label: str):
        self.filename = filename
        self.label = label

    def serialize(self) -> dict:
        return {
            'filename': self.filename,
            'label': self.label
        }


class SamplePair:
    def __init__(self, first_sample: DatasetItem, second_sample: DatasetItem, same_label: bool):
        self.first_sample = first_sample
        self.second_sample = second_sample
        self.same_label = same_label

    def serialize(self) -> dict:
        return {
            'first_sample': self.first_sample.serialize(),
            'second_sample': self.second_sample.serialize(),
            'same_label': self.same_label
        }


class SampleLogger:
    def __init__(self, dump_file: str):
        self._dump_file = dump_file
        self._sample_pairs = []
        self._same_label_count = 0
        self._total_count = 0

    def add_sample_pair(self, first_item: DatasetItem, second_item: DatasetItem, same_label: bool):
        sample_pair = SamplePair(first_item, second_item, same_label)
        self._sample_pairs.append(sample_pair)
        if same_label:
            self._same_label_count += 1
        self._total_count += 1

    def serialize(self) -> list:
        return [sample_pair.serialize() for sample_pair in self._sample_pairs]

    def save(self):
        dump_filepath = Path(self._dump_file)
        try:
            samples = self.serialize()
            with open(dump_filepath, 'w') as out_file:
                json.dump(samples, out_file)
            print(f'Samples log saved to {dump_filepath}')
        except Exception as exception:
            print(f'An error occured while trying to write samples to file.', exception)