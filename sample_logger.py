from typing import Dict


class DatasetItem:
    def __init__(self, filename, label):
        self.filename = filename
        self.label = label


class SamplePair:
    def __init__(self, first_sample: DatasetItem, second_sample: DatasetItem, same_label: bool):
        self.first_sample = first_sample
        self.second_sample = second_sample
        self.same_label = same_label


class LabelStatistics:
    def __init__(self, label):
        self._label = label
        self._samples_taken = 0
        self._samples_used = {}

    @property
    def label(self):
        return self._label

    @property
    def samples_taken(self):
        return self._samples_taken

    @property
    def samples_used(self):
        return self._samples_used

    def add_sample(self, sample: DatasetItem):
        self._samples_used[sample.filename] += 1


class SampleLogger:
    def __init__(self):
        self._sample_pairs = []
        self._same_label_count = 0
        self._total_count = 0

    def add_sample_pair(self, first_item: DatasetItem, second_item: DatasetItem, same_label: bool):
        sample_pair = SamplePair(first_item, second_item, same_label)
        self._sample_pairs.append(sample_pair)
        if same_label:
            self._same_label_count += 1
        self._total_count += 1

    def get_statistics(self) -> Dict[int, LabelStatistics]:
        label_to_samples_map = {}
        for sample_pair in self._sample_pairs:
            first_item_label = sample_pair.first_sample.label
            if first_item_label not in label_to_samples_map:
                label_to_samples_map[first_item_label] = LabelStatistics(first_item_label)

            label_to_samples_map[first_item_label].add_sample(sample_pair.first_sample)

        return label_to_samples_map