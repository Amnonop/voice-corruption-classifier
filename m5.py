import torch.nn as nn


class M5(nn.Module):
    def __init__(self, num_classes):
        super(M5, self).__init__()

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
        self.softmax_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Global avg pooling
        x = self.avg_pool(x) # [batch_size, 256, 1]

        # Dense
        x = x.view(x.size(0), -1)  # [batch_size, 256*1=256]
        x = self.softmax_layer(x)  # [batch_size, 10]
        return x
