import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AudioDataset import AudioDataset, TrainTestSplitter
from pathlib import Path


###https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py###
##download dataset and extract

#transform = transforms.Compose(
 #   [transforms.ToTensor(),
  #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
csv_filepath = Path('data/clean_speech/spkrinfo.csv');
train_test_splitter = TrainTestSplitter(csv_file=csv_filepath, test_ratio=0.2)

trainset = AudioDataset(train_test_splitter, csv_file=csv_filepath, root_dir='data/clean_speech', is_train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = AudioDataset(train_test_splitter, csv_file=csv_filepath, root_dir='data/clean_speech', is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('M', 'F')



#showing an image for fun:

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training files
dataiter = iter(trainloader)
samples = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[samples['label'][j]] for j in range(4)))

#define the nn:

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Taken from the paper M3 (?)
        self.conv1 = nn.Conv1d(in_channels=10000, out_channels=256, kernel_size=(80,1), stride=(4,1))
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=(3,1), stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#ֳoptimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#define loss function:

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#network training:


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
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
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# save net state dict
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#test vs ground truth

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#privaate testing:

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#performence on all dataset:

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#accuracy analysis per category:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


#for i in range(10):
#    print('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]))












