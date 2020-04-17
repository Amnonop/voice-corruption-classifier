from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from commons import CONFIGS_DIR
from configuration import *
from dataset_transforms import TransformsComposer, ToTensor, Rescale
from classifier import Classifier
from data_loader import DataLoader
from m5 import M5
from audio_dataset import AudioDataset

CONFIG_FILENAME = 'config.json'

def main():
    config_filename = Path.cwd().joinpath(CONFIGS_DIR).joinpath(CONFIG_FILENAME)
    config = Configuration(config_filename)

    csv_path = Path(config.csv_filename)
    if not csv_path.is_file():
        raise Exception(f'{csv_path} does not exist.')
    data_dir = Path(config.data_dir)
    if not Path(data_dir).is_dir():
        raise Exception(f'{data_dir} does not exist.')

    batch_size = 4
    epochs = 32

    model = M5(num_classes=4)
    classifier = Classifier(model=model, state_path=f'./state_{epochs}_epochs_1.pth')

    transforms = TransformsComposer([Rescale(output_size=10000), ToTensor()])

    # Split to train and test
    # train_set, test_set = data_loader.split(test_ratio=0.2)
    train_csv_path = Path(config.train_csv)
    train_set = pd.read_csv(train_csv_path, header=0)

    # Create the classes based on configuration
    class_column_index = train_set.columns.get_loc(config.classify_by)
    class_column = train_set.iloc[:, class_column_index]
    encoder = LabelEncoder()
    classes_encoded = encoder.fit_transform(class_column)
    classes = encoder.classes_
    classes_map = {}
    for i, category in enumerate(classes):
        classes_map[category] = i
    print(classes_map)

    train_dataset = AudioDataset(train_set, data_dir, transforms)

    test_csv_path = Path(config.test_csv)
    test_set = pd.read_csv(test_csv_path, header=0)
    test_dataset = AudioDataset(test_set, data_dir, transforms)

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

if __name__ == '__main__':
    main()