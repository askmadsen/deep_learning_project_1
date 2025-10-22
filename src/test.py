from cat_dog_classifier import CatDogClassifier
from cat_dog_dataset import CatDogDataset
from cat_dog_classifier_good import CatDogClassifierGood
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from data_handler import predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID = "42"
MODEL_PATH = f"models/cat_dog_model_{ID}.pth"



def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = loss / len(test_loader)
    return accuracy, avg_loss




if __name__ == "__main__":
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

   
    classifier = CatDogClassifierGood()
    classifier.load_state_dict(torch.load(MODEL_PATH))
    classifier.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    acc, loss = test_model(classifier, test_loader, criterion, DEVICE)
    
    print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")
    predict(classifier, "catdog_data/test/cats/cat.1301.jpg", transformations, DEVICE)

    