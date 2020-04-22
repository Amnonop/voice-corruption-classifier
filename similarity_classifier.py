import time
import copy
import sys
from typing import Tuple

from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
import torch
import pandas as pd

from commons import *


class SimilarityClassifier:
    def __init__(self, model, state_path):
        self.model = model
        self.state_path = state_path

    def train(self, data_loader, optimizer, criterion) -> Tuple[float, float]:
        train_loss = 0.0
        correct_count = 0
        total = 0

        for first_sample, second_sample, label in data_loader:
            batch_size = label.size(0)

            # switch model to training mode
            self.model.train()

            # clear gradient accumulators
            optimizer.zero_grad()

            # forward pass
            output = self.model(first_sample['signal'], second_sample['signal'])

            # calculate loss of the network output with respect to the training labels
            loss = criterion(output, label)

            # backpropagate and update optimizer learning rate
            loss.backward()
            optimizer.step()

            # Statistics
            correct_count += (torch.max(output, 1)[1].view(label.size()) == label).sum().item()
            train_loss += (loss.item() / batch_size)
            total += batch_size

        train_accuracy = 100. * correct_count / total
        return train_loss, train_accuracy

    def validate(self, validation_data, batch_size, optimizer, criterion) -> Tuple[float, float]:
        data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

        validation_loss = 0.0
        correct_count = 0
        total = 0

        # switch model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for first_sample, second_sample, label in data_loader:
                batch_size = label.size(0)

                output = self.model(first_sample['signal'], second_sample['signal'])

                loss = criterion(output, label)

                # Statistics
                correct_count += (torch.max(output, 1)[1].view(label.size()) == label).sum().item()
                validation_loss += (loss.item() / batch_size)
                total += batch_size

        validation_accuracy = 100. * correct_count / total
        return validation_loss, validation_accuracy

    def fit(self, train_set, batch_size, epochs, validation_data, verbose=False, shuffle=True):
        train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

        since = time.time()

        val_acc_history = []
        train_loss_history = []
        val_loss_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        learning_rate = 0.001
        optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        loss_function = BCEWithLogitsLoss(size_average=True)

        # Train the network
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            train_loss, train_accuracy = self.train(train_data_loader, optimizer, loss_function)

            #epoch_loss = train_loss / len(train_data_loader.dataset)
            #epoch_acc = train_accuracy.double() / len(train_data_loader.dataset)

            train_loss_history.append(train_loss)

            print('Train Loss: {:.5f} Acc: {:.3f}'.format(train_loss, train_accuracy))

            validation_loss, validation_accuracy = self.validate(validation_data, batch_size, optimizer, loss_function)

            #epoch_loss = validation_loss / len(validation_data)
            #epoch_acc = validation_accuracy.double() / len(validation_data)

            print('Validation Loss: {:.5f} Acc: {:.3f}'.format(validation_loss, validation_accuracy))

            # update best weights
            # TODO: save a snapshot at this point
            if validation_accuracy > best_acc:
                best_acc = validation_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())

            val_acc_history.append(validation_accuracy)
            val_loss_history.append(validation_loss)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
        print(f'Best val Acc: {best_acc}')

        # Load best model weights and save them
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), self.state_path)

        return train_loss_history, val_loss_history

    def save_predictions(self, predictions, filepath):
        data_frame = pd.DataFrame(predictions, columns=['filename 1', 'filename 2', 'label', 'predicted'])
        data_frame.to_csv(filepath, index=False)

    def predict(self, test_set, batch_size, output_filepath):
        data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        self.model.load_state_dict(torch.load(self.state_path))
        correct = 0.0
        total = 0.0

        classes = {
            0: 'different',
            1: 'same'}

        num_classes = len(classes.keys())
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        predictions = []

        start_time = time.time()

        print('Starting prediction')

        with torch.no_grad():
            for first_sample, second_sample, targets in data_loader:
                outputs = self.model(first_sample['signal'], second_sample['signal'])
                predicted = torch.max(outputs.data, 1)[1]

                for i in range(len(targets)):
                    class_id = int(targets[i].item())
                    if predicted[i] == class_id:
                        class_correct[class_id] += 1
                    class_total[class_id] += 1

                    # Save prediction in format of file_1, file_2, label, predicted
                    predictions.append((first_sample['filename'][i], second_sample['filename'][i], class_id, predicted[i].item()))

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        time_elapsed = time.time() - start_time
        print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Save predictions
        self.save_predictions(predictions, output_filepath)

        print('Accuracy of the network on the test set: {:.2f} %'.format(100. * sum(class_correct) / total))

        for i in range(len(classes)):
            print('Accuracy of {:.9s} : {:.2f} %'.format(
                classes[i], 100 * class_correct[i] / class_total[i]))
