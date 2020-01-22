import torch
import torch.nn as nn
import torch.optim as optim
from AudioDataset import AudioDataset, TrainTestSplitter
from pathlib import Path
from nn_modules import Net
from matplotlib import pyplot as plt

###https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py###
##download dataset and extract

#transform = transforms.Compose(
 #   [transforms.ToTensor(),
  #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
csv_filepath = Path('data/clean_speech/spkrinfo.csv');
train_test_splitter = TrainTestSplitter(csv_file=csv_filepath, test_ratio=0.2)

trainset = AudioDataset(train_test_splitter, csv_file=csv_filepath, root_dir='data/clean_speech', is_train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = AudioDataset(train_test_splitter, csv_file=csv_filepath, root_dir='data/clean_speech', is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('M', 'F')

#define the nn:
net = Net()


# define loss function:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# network training:
num_epochs = 16
train_loss_values = []
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['audio']
        labels = data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, epoch_loss / 20))
            epoch_loss = 0.0

    # Save loss value for current epoch run
    train_loss_values.append(running_loss / len(trainset))

print('Finished Training')
plt.title(f'Model Loss for {num_epochs} epochs')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_loss_values)
plt.show()

# save net state dict
PATH = './speech_net.pth'
torch.save(net.state_dict(), PATH)

# test vs ground truth

dataiter = iter(testloader)
data = dataiter.next()
test_inputs = data['audio']
test_labels = data['label']

net = Net()
net.load_state_dict(torch.load(PATH))

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
        for i in range(4):
            label = test_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))












