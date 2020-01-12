import torch
from AudioDataset import AudioDataset, TrainTestSplitter
from pathlib import Path
import torch.nn as nn
from nn_modules import Net

classes = ('M', 'F')

csv_filepath = Path('data/mf_test/spkrinfo.csv')
train_test_splitter = TrainTestSplitter(csv_file=csv_filepath, test_ratio=0.9)

testset = AudioDataset(train_test_splitter, csv_file=csv_filepath, root_dir='data/mf_test', is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

# Path to the net's state dictionary
STATE_DICT_PATH = './speech_net.pth'

# test vs ground truth
dataiter = iter(testloader)
data = dataiter.next()
test_inputs = data['audio']
test_labels = data['label']

net = Net()
net.load_state_dict(torch.load(STATE_DICT_PATH))

outputs = net(test_inputs)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#performence on all dataset:

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        test_inputs = data['audio']
        test_labels = data['label']
        outputs = net(test_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

print('Accuracy of the network on the test set: %d %%' % (
    100 * correct / total))

# Accuracy analysis per category:
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        test_inputs = data['audio']
        test_labels = data['label']
        outputs = net(test_inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == test_labels).squeeze()
        for i in range(len(test_labels)):
            label = test_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))