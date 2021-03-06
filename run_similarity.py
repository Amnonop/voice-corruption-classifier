import uuid
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from commons import CONFIGS_DIR, STATES_DIR
from configuration import *
from dataset_transforms import TransformsComposer, ToTensor, Rescale
from classifier import Classifier
from data_loader import DataLoader
from m5 import M5
from siamese import Siamese
from similarity_classifier import SimilarityClassifier
from similarity_dataset import SimilarityDataset
from utils import create_results_directories
from sample_logger import SampleLogger

CONFIG_FILENAME = 'config.json'
RESULTS_DIR = 'siamese_results/'
SAMPLE_LOGGER_FILE = 'samples.json'


def main():
    config_filename = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = Configuration(config_filename)

    batch_size = 4
    epochs = 4

    results_dir_path = Path.cwd().joinpath(RESULTS_DIR)
    current_run_path = create_results_directories(results_dir_path)

    sample_logger_path = Path.cwd().joinpath(current_run_path).joinpath(SAMPLE_LOGGER_FILE)
    sample_logger = SampleLogger(sample_logger_path)

    transforms = TransformsComposer([Rescale(output_size=10000), ToTensor()])

    encoder = LabelEncoder()

    data_loader = DataLoader(config)
    x_train, y_train = data_loader.get_train_set()
    encoder.fit(y_train)

    classes = encoder.classes_
    classes_map = {}
    for i, category in enumerate(classes):
        classes_map[i] = category
    print(classes_map)

    y_train = encoder.transform(y_train)
    train_dataset = SimilarityDataset(x_train, y_train, classes_map, sample_logger, transforms)

    x_test, y_test = data_loader.get_test_set()
    y_test = encoder.transform(y_test)
    test_dataset = SimilarityDataset(x_test, y_test, classes_map, sample_logger, transforms)

    model = Siamese()

    states_dir = Path.cwd().joinpath(STATES_DIR)
    state_filename = f'{uuid.uuid1()}_state_{epochs}_epochs.pth'
    state_path = current_run_path.joinpath('best_snapshot').joinpath(state_filename)

    classifier = SimilarityClassifier(model=model, state_path=state_path)

    # Fit model on data
    train_loss_history, val_loss_history = classifier.fit(train_dataset, batch_size=batch_size, epochs=epochs,
                                                          validation_data=test_dataset)

    sample_logger.save()

    # plt.figure()
    # plt.title(f'Model Loss for {epochs} epochs')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(train_loss_history, label='train')
    # plt.plot(val_loss_history, label='test')
    # plt.legend()
    # plt.show()

    predictions_path = Path.cwd().joinpath('./predicted.csv')
    validation_dataset = SimilarityDataset(x_test, y_test, classes_map, sample_logger, transforms)
    validation_model = Siamese(num_classes=len(classes_map))
    validation_classifier = SimilarityClassifier(validation_model, state_path=state_path)
    validation_classifier.predict(validation_dataset, batch_size=batch_size, output_filepath=predictions_path)


if __name__ == '__main__':
    main()