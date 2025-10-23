from torch import nn
import torch

class CatDogClassifierNew(nn.Module):
    
    def __init__(self):
        super(CatDogClassifierNew, self).__init__()  # [batch_size, 3, 128, 128]

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8

           
    

        )
       
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 2)


        
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)

        # Flatten input tensor
        x = torch.flatten(x, 1)

        # Classifier
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

