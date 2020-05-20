import torch.nn as nn
import torch


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=80, stride=4),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.avg_pool = nn.AvgPool1d(9)
        #self.softmax_layer = nn.Linear(512, num_classes)

        self.linear = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())

        self.out = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())


    def forward_one(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Global avg pooling
        x = self.avg_pool(x)  # [batch_size, 256, 1]

        # Dense
        x = x.view(x.size(0), -1)  # [batch_size, 256*1=256]
        x = self.linear(x)  # [batch_size, 10]
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out




if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))