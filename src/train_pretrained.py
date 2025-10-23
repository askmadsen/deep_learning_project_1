import torch
import torch.nn as nn
from torchvision import models
from train import train_epoch
from cat_dog_dataset import CatDogDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from test import test_model
from data_handler import predict

EPOCHS = 5
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
ID = "5"
MODEL_PATH = f"models/cat_dog_model_{ID}.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FN = torch.nn.CrossEntropyLoss()
TUNE = False






def get_pretrained_model() -> nn.Module:
    vgg = models.vgg16(pretrained=True)

    for param in vgg.features.parameters():
        param.requires_grad = False

    vgg.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 2),
    )

    print(vgg.features)
    return vgg

def load_pretrained_model(model_path: str) -> nn.Module:
    vgg = models.vgg16(pretrained=False)  # pretrained=False since we are loading saved weights

    for param in vgg.features.parameters():
        param.requires_grad = False

    vgg.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 2),
    )


    vgg.load_state_dict(torch.load(model_path))
    return vgg


if __name__ == "__main__":
    get_pretrained_model()
    """if TUNE:
        transformations_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
        
        transformations_validation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = CatDogDataset(dogs_dir="catdog_data/train/dogs", cats_dir="catdog_data/train/cats", transform=transformations_train)
        validation_dataset = CatDogDataset(dogs_dir="catdog_data/validation/dogs", cats_dir="catdog_data/validation/cats", transform=transformations_validation)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        classifier = get_pretrained_model()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        run_epoch(classifier, train_dataloader, LOSS_FN, optimizer, DEVICE)
        train_accuracies = []
        train_losses = []
        validation_accuracies = []
        validation_losses = []

        for epoch in range(EPOCHS):
            train_acc, train_loss = run_epoch(classifier, train_dataloader, LOSS_FN, optimizer, DEVICE)
            val_acc, val_loss = test_model(classifier, validation_dataloader, LOSS_FN, DEVICE)
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            validation_accuracies.append(val_acc)
            validation_losses.append(val_loss)
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    else:
        classifier = load_pretrained_model(MODEL_PATH)
        classifier.to(DEVICE)
        transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

        test_dataset = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformations)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        acc, loss = test_model(classifier, test_loader, LOSS_FN, DEVICE)
        
        print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")
        predict(classifier, "catdog_data/test/cats/cat.1300.jpg", transformations, DEVICE)
"""