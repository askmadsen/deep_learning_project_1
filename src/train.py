from cat_dog_dataset import CatDogDataset
from cat_dog_classifier import CatDogClassifier
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from test import test_model
from data_handler import save_data, plot_metrics

EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FN = torch.nn.CrossEntropyLoss()
MODEL_PATH = "models/cat_dog_model.pth"
DATA_PATH = "training_metrics.json"
IMAGE_PATH = "training_metrics.png"


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
    avg_loss = running_loss / len(train_dataloader)

    
    torch.save(model.state_dict(), MODEL_PATH)

    return accuraccy, avg_loss
    




if __name__ == "__main__":
    transformations_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation((-90, 90)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])  
    transformations_validation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

    train_dataset = CatDogDataset(dogs_dir="catdog_data/train/dogs", cats_dir="catdog_data/train/cats", transform=transformations_train)
    validation_dataset = CatDogDataset(dogs_dir="catdog_data/validation/dogs", cats_dir="catdog_data/validation/cats", transform=transformations_validation)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    classifier = CatDogClassifier()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
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
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    save_data(DATA_PATH, {
        "train_acc": train_accuracies,
        "train_loss": train_losses,
        "validation_acc": validation_accuracies,
        "validation_loss": validation_losses
    })

    plot_metrics(IMAGE_PATH, DATA_PATH)

