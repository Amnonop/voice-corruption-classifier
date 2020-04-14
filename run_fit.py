from pathlib import Path

from configuration import *
from dataset_transforms import TransformsComposer, ToTensor
from classifier import Classifier
from data_loader import DataLoader
from m5 import M5

CONFIG_FILENAME = 'configs/config.json'

def main():
    config = Configuration(CONFIG_FILENAME)

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

    transform = TransformsComposer([ToTensor()])

    data_loader = DataLoader(csv_path)

    # Split to train and test
    train_set, test_set = data_loader.split(test_ratio=0.2)
    train_dataset = ECGDataset(train_set, data_dir, transform)
    test_dataset = ECGDataset(test_set, data_dir, transform)

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