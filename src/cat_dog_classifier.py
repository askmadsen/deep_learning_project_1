import torch.nn as nn

class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()                                                    # [batch_size, 3, 128, 128]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)  # [batch_size, 16, 126, 126]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                # [batch_size, 16, 63, 63]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0) # [batch_size, 32, 61, 61]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                               # [batch_size, 32, 30, 30]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0) # [batch_size, 64, 28, 28]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                               # [batch_size, 64, 14, 14]

        self.fc1 = nn.Linear(14 * 14 * 64, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        # Convolutional part
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Flatten input tensor
        x = x.view(x.size(0), -1)

        # Connected part
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x