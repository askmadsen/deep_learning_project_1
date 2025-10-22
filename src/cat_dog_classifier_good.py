from torch import nn

class CatDogClassifierGood(nn.Module):
    
    def __init__(self):
        super(CatDogClassifierGood, self).__init__()                                                # [batch_size, 3, 128, 128]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # [batch_size, 32, 128, 128]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                # [batch_size, 32, 64, 64]

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # [batch_size, 64, 64, 64]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                               # [batch_size, 64, 32, 32]

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # [batch_size, 128, 32, 32]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                # [batch_size, 128, 16, 16]
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # [batch_size, 128, 16, 16]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                # [batch_size, 128, 8, 8]

        self.conv_dropout = nn.Dropout2d(0.2)

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))


        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)


        # ID 5 is current best model with pool(4,4) and fc1 = 256
    
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
        x = self.conv_dropout(x)


        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.conv_dropout(x)

        x = self.avgpool(x)



        # Flatten input tensor
        x = x.view(x.size(0), -1)

        # Connected part
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x