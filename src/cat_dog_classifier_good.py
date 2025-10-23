from torch import nn

class CatDogClassifierGood(nn.Module):
    
    def __init__(self):
        super(CatDogClassifierGood, self).__init__()  # [batch_size, 3, 128, 128]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # [batch_size, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 32, 64, 64]

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # [batch_size, 64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 64, 32, 32]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # [batch_size, 128, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 128, 16, 16]
            nn.Dropout2d(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # [batch_size, 256, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # [batch_size, 256, 8, 8]
            nn.Dropout2d(0.2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )

        # ID 5 is current best model with pool(4,4) and fc1 = 256
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)

        # Flatten input tensor
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.classifier(x)
        return x
    

