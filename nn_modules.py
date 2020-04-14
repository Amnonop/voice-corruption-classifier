import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Taken from the paper M3 (?)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.avg_pool = nn.AvgPool1d(154)
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Global avg pooling
        x = self.avg_pool(x) # [batch_size, 256, 1]

        # Dence
        x = x.view(x.size(0), -1) # [batch_size, 256*1=256]
        x = self.out(x) # [batch_size, 10]
        return x




class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
        # Taken from the paper M3 (?)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        self.avg_pool = nn.AvgPool1d(9)
        self.out = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # Global avg pooling
        x = self.avg_pool(x) # [batch_size, 256, 1]

        # Dence
        x = x.view(x.size(0), -1) # [batch_size, 256*1=256]
        x = self.out(x) # [batch_size, 10]
        return x
