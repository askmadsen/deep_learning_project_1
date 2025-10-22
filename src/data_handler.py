import json
from typing import Type
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def save_data(data_path: str, data):
    with open(data_path, "w") as f:
        json.dump(data, f, indent=4)
    
def load_data(data_path: str):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def plot_metrics(save_path: str, data_path: str):
    data = load_data(data_path)

    train_losses = data["train_loss"]
    validation_losses = data["validation_loss"]
    train_accuracies = data["train_acc"]
    test_accuracies = data["validation_acc"]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def predict(model: nn.Module, image_path: str, transform: transforms.Compose, device: torch.device):
    # Set model to evaluation mode
    model.eval()
    model.to(device)

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    # Convert confidence to percentage
    confidence_percent = confidence.item() * 100
    predicted_label = "Dog" if predicted.item() == 0 else "Cat"

    # Show the image with predicted label and confidence
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label} ({confidence_percent:.2f}%)")
    plt.axis('off')
    plt.show()
