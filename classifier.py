import time
import copy

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch
import pandas as pd

from commons import *


class Classifier:
    def __init__(self, model, state_path):
        self.model = model
        self.state_path = state_path

    def train(self, data_loader, optimizer, criterion):
        running_loss = 0.0
        running_corrects = 0

        for data in data_loader:
            inputs = data['ecg']
            labels = data['label']

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects

    def validate(self, validation_data, batch_size, optimizer, criterion):
        data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

        running_loss = 0.0
        running_corrects = 0

        for data in data_loader:
            inputs = data['ecg']
            labels = data['label']

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects

    def fit(self, train_set, batch_size, epochs, validation_data, verbose=False, shuffle=True):
        test_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

        since = time.time()

        val_acc_history = []
        train_loss_history = []
        val_loss_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        learning_rate = 0.001
        optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        loss_function = CrossEntropyLoss()

        # Train the network
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            running_loss, running_corrects = self.train(test_set_loader, optimizer, loss_function)

            epoch_loss = running_loss / len(test_set_loader.dataset)
            epoch_acc = running_corrects.double() / len(test_set_loader.dataset)

            train_loss_history.append(epoch_loss)

            print(f'Train Loss: {epoch_loss} Acc: {epoch_acc}')

            running_loss, running_corrects = self.validate(validation_data, batch_size, optimizer, loss_function)

            epoch_loss = running_loss / len(test_set_loader.dataset)
            epoch_acc = running_corrects.double() / len(test_set_loader.dataset)

            print(f'Validation Loss: {epoch_loss} Acc: {epoch_acc}')

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            val_acc_history.append(epoch_acc)
            val_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
        print(f'Best val Acc: {best_acc}')

        # Load best model weights and save them
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), self.state_path)

        return train_loss_history, val_loss_history

    def save_predictions(self, predictions, filepath):
        data_frame = pd.DataFrame(predictions, columns=['Filename', 'Class', 'Predicted'])
        data_frame.to_csv(filepath)

    def predict(self, test_set, batch_size, output_filepath):
        data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        self.model.load_state_dict(torch.load(self.state_path))
        correct = 0.0
        total = 0.0

        num_classes = len(class_ids.keys())
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        predictions = []

        start_time = time.time()

        with torch.no_grad():
            for data in data_loader:
                filenames = data['filename']
                inputs = data['ecg']
                targets = data['label']

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                c = (predicted == targets).squeeze()
                for i in range(len(targets)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                    # Save prediction in format of file, label, predicted
                    predictions.append((filenames[i], get_class_name(label), get_class_name(predicted[i])))

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        time_elapsed = time.time() - start_time
        print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Save predictions
        self.save_predictions(predictions, output_filepath)

        print('Accuracy of the network on the test set: %d %%' % (
                100 * correct / total))

        for i in range(4):
            print('Accuracy of %5s : %2d %%' % (
                get_class_name(i), 100 * class_correct[i] / class_total[i]))