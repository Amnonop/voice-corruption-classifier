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
from audio_dataset import AudioDataset
from utils import create_results_directories

CONFIG_FILENAME = 'config.json'
RESULTS_DIR = 'results/'
SAMPLE_LOGGER_FILE = 'samples.json'

def main():
    config_filename = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = Configuration(config_filename)

    batch_size = 4
    epochs = 1

    results_dir_path = Path.cwd().joinpath(RESULTS_DIR)
    current_run_path = create_results_directories(results_dir_path)

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
    train_dataset = AudioDataset(x_train, y_train, transforms)

    x_test, y_test = data_loader.get_test_set()
    y_test = encoder.transform(y_test)
    test_dataset = AudioDataset(x_test, y_test, transforms)

    model = M5(num_classes=len(classes_map))

    states_dir = Path.cwd().joinpath(STATES_DIR)
    state_filename = f'{uuid.uuid1()}_state_{epochs}_epochs.pth'
    state_path = current_run_path.joinpath('best_snapshot').joinpath(state_filename)

    classifier = Classifier(model=model, state_path=state_path)

    # Fit model on data
    train_loss_history, val_loss_history = classifier.fit(train_dataset, batch_size=batch_size, epochs=epochs,
                                                          validation_data=test_dataset)

    plt.figure()
    plt.title(f'Model Loss for {epochs} epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='test')
    plt.legend()
    plt.show()
    
    predictions_path = current_run_path.joinpath('./predicted.csv')
    validation_dataset = AudioDataset(x_test, y_test, transforms)
    validation_model = M5(num_classes=len(classes_map))
    validation_classifier = Classifier(validation_model, state_path=state_path)
    validation_classifier.predict(validation_dataset, batch_size=batch_size, output_filepath=predictions_path, classes=classes_map)


if __name__ == '__main__':
    main()