from cat_dog_classifier import CatDogClassifier
from cat_dog_classifier_good import CatDogClassifierGood
from cat_dog_classifier_new import CatDogClassifierNew
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from test import test_model
from data_handler import save_data, plot_metrics
from config import TRAIN_DATA as train_dataset
from config import VALIDATION_DATA as validation_dataset
from config import MODEL_DIR, TRAINING_METRICS_DIR, TRAINING_METRIC_IMAGES_DIR


EPOCHS = 50 
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_FN = torch.nn.CrossEntropyLoss()

def train_epoch(model: torch.nn.Module, train_loader: DataLoader, criterion, optimizer, device: torch.device):
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

    return accuraccy, avg_loss
    
def set_seed(seed=42):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(num_epochs: int,
                model: torch.nn.Module,
                train_loader: DataLoader,
                validation_loader: DataLoader,
                criterion,
                device: torch.device,
                param_save_path: str | None = None,
                metric_data_path: str | None = None,
                lock_convolutional_layers: bool = False
                ) -> None:

    #classifier = CatDogClassifier()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # monitor val_acc
        factor=0.8,           # slightly stronger decay per trigger
        patience=3,           # fewer epochs to wait before decaying
        threshold=0.01,       # require at least 1% improvement in val_acc
        threshold_mode='rel', # relative improvement
        min_lr=1e-6,          # never go below this LR
        cooldown=2            # wait 2 epochs after LR decay before monitoring
    )


 

    train_accuracies = []
    train_losses = []
    validation_accuracies = []
    validation_losses = []

    for epoch in range(num_epochs):

        # Train for one epoch
        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_acc, val_loss = test_model(model, validation_loader, LOSS_FN, DEVICE)

        # Record metrics
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        validation_accuracies.append(val_acc)
        validation_losses.append(val_loss)

        # Step the scheduler
        scheduler.step(val_acc)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics + LR
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}")

    if (param_save_path is not None):
        torch.save(model.state_dict(), param_save_path)

    if (metric_data_path is not None):
        save_data(metric_data_path, {
            "train_acc": train_accuracies,
            "train_loss": train_losses,
            "validation_acc": validation_accuracies,
            "validation_loss": validation_losses
        })

if __name__ == "__main__":
    #seeds = [42, 1054, 1406, 3772, 4609, 5168, 6525, 7214, 8263, 9210]

    seeds = [6525, 4609, 1406]
    for seed in seeds:
        print(seed)
        set_seed(seed)

        lock_features = False

        model = CatDogClassifierNew()

        if lock_features:
            for param in model.features.parameters():
                param.requires_grad = False

        train_model(
            num_epochs=EPOCHS,
            model=model,
            train_loader=DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            validation_loader=DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False),
            criterion=LOSS_FN,
            device=DEVICE,
            param_save_path=f"{MODEL_DIR}good_{seed}.pth",
            metric_data_path=f"{TRAINING_METRICS_DIR}good_{seed}.json"
        )

