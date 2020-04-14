import torch
import numpy as np


class TransformsComposer(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, signal):
        original_length = len(signal)

        if original_length < self.output_size:
            signal = np.concatenate((signal, np.zeros(self.output_size - original_length)))
        elif original_length > self.output_size:
            signal = signal[0:self.output_size]

        return signal


class ToTensor(object):
    def __call__(self, signal):
        signal_reshape = signal.reshape(1, -1)
        signal = torch.tensor(signal_reshape, dtype=torch.float)
        return signal