from cat_dog_classifier import CatDogClassifier
from cat_dog_dataset import CatDogDataset
import torch
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "cat_dog_model.pth"

transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation((-90, 90)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformations)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
classifier = CatDogClassifier()
classifier.load_state_dict(torch.load(MODEL_PATH))

classifier.to(DEVICE)
classifier.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Accuraccy of the model on the test images: {accuracy:.2f}%")
