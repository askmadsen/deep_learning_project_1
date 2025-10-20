from cat_dog_dataset import CatDogDataset
from cat_dog_classifier import CatDogClassifier
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FN = torch.nn.CrossEntropyLoss()
MODEL_PATH = "cat_dog_model.pth"

transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation((-90, 90)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = CatDogDataset(dogs_dir="catdog_data/train/dogs", cats_dir="catdog_data/train/cats", transform=transformations)
validation_dataset = CatDogDataset(dogs_dir="catdog_data/validation/dogs", cats_dir="catdog_data/validation/cats", transform=transformations)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

classifier = CatDogClassifier()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)


classifier.to(DEVICE)
classifier.train()

for epoch in range(EPOCHS):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_dataloader:

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = classifier(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuraccy = correct / total
    avg_loss = running_loss / len(train_dataloader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuraccy:.4f}")

torch.save(classifier.state_dict(), MODEL_PATH)
