from cat_dog_classifier import CatDogClassifier
from cat_dog_dataset import CatDogDataset
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/cat_dog_model.pth"



def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> float:
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
    predicted_label = "Dog" if predicted.item() == 1 else "Cat"

    # Show the image with predicted label and confidence
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label} ({confidence_percent:.2f}%)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

    test_dataset = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    classifier = CatDogClassifier()
    classifier.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.CrossEntropyLoss()
    acc, loss = test_model(classifier, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")
    predict(classifier, "catdog_data/test/cats/cat.1300.jpg", transformations, DEVICE)

    