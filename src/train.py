from cat_dog_dataset import CatDogDataset
from cat_dog_classifier import CatDogClassifier
from cat_dog_classifier_good import CatDogClassifierGood
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from test import test_model
from data_handler import save_data, plot_metrics

EPOCHS = 150
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FN = torch.nn.CrossEntropyLoss()

ID = "42"
MODEL_PATH = f"models/cat_dog_model_{ID}.pth"
DATA_PATH = f"training_metrics_{ID}.json"
IMAGE_PATH = f"training_metrics_{ID}.png"


def train_model(model: torch.nn.Module, train_loader: DataLoader, criterion, optimizer, device: torch.device):
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuraccy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    
    torch.save(model.state_dict(), MODEL_PATH)

    return accuraccy, avg_loss
    
def set_seed(seed=42):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    
    set_seed(int(ID))
    
    transformations_train = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8,1.0)),   # more robust than Resize + CenterCrop
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])
    
    transformations_validation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CatDogDataset(dogs_dir="catdog_data/train/dogs", cats_dir="catdog_data/train/cats", transform=transformations_train)
    validation_dataset = CatDogDataset(dogs_dir="catdog_data/validation/dogs", cats_dir="catdog_data/validation/cats", transform=transformations_validation)
    test_dataset = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformations_validation)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #classifier = CatDogClassifier()
    classifier = CatDogClassifierGood()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',           # monitor val_acc
    factor=0.9,           # smaller decay per trigger (more gentle)
    patience=10,          # wait longer before decaying
    threshold=0.005,      # only consider improvement significant if val_acc increases by >= 0.5%
    threshold_mode='rel', # relative threshold
    min_lr=1e-6,          # stop decaying too low
    cooldown=2            # wait 2 epochs after a decay before monitoring again
)

    
    train_accuracies = []
    train_losses = []
    validation_accuracies = []
    validation_losses = []

    for epoch in range(EPOCHS):
        train_acc, train_loss = train_model(classifier, train_dataloader, LOSS_FN, optimizer, DEVICE)
        val_acc, val_loss = test_model(classifier, validation_dataloader, LOSS_FN, DEVICE)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        validation_accuracies.append(val_acc)
        validation_losses.append(val_loss)
        scheduler.step(val_acc)


        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics + LR
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}")

        # Optional: evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_acc, test_loss = test_model(classifier, test_dataloader, LOSS_FN, DEVICE)
            print(f"--- Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% ---")

    save_data(DATA_PATH, {
        "train_acc": train_accuracies,
        "train_loss": train_losses,
        "validation_acc": validation_accuracies,
        "validation_loss": validation_losses
    })

    plot_metrics(IMAGE_PATH, DATA_PATH)

