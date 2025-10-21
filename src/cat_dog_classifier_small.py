import torch.nn as nn

class CatDogClassifierSmall(nn.Module):
    def __init__(self):
        super(CatDogClassifierSmall, self).__init__()                                                    # [batch_size, 3, 256, 256]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)  # [batch_size, 16, 252, 252]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                # [batch_size, 16, 126, 126]

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # [batch_size, 32, 122, 122]
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)                               # [batch_size, 32, 40, 40]

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0) # [batch_size, 32, 38, 38]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                               # [batch_size, 32, 19, 19]
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0) # [batch_size, 32, 17, 17]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                               # [batch_size, 32, 8, 8]

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
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

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        # Flatten input tensor
        x = x.view(x.size(0), -1)

        # Connected part
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x