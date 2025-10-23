import json
import math
import random
from typing import Type
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
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



def plot_confusion_matrix(model: nn.Module,
                          model_path: str,
                          device: torch.device,
                          test_loader,
                          class_names: list = ["Dog", "Cat"]):
    """
    Predicts on a test DataLoader and plots a normalized confusion matrix.
    """

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.show()


def _predict_image(model: nn.Module, image_tensor, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return predicted.item(), confidence.item()


def predict_single(model: nn.Module, model_path: str, image_path: str, transform, device: torch.device, class_names=["Dog", "Cat"]):
    model.load_state_dict(torch.load(model_path, map_location=device))
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    predicted_idx, confidence = _predict_image(model, image_tensor, device)
    confidence_percent = confidence * 100
    predicted_label = class_names[predicted_idx]

    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label} ({confidence_percent:.2f}%)")
    plt.axis('off')
    plt.show()

def denormalize(tensor, mean, std):
    """
    Undo normalization to recover the original image range [0,1]
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def predict_grid_dataset(model: nn.Module, model_path: str, dataset, device: torch.device , n_per_class: int = 5, class_names=["Dog", "Cat"]):
    """
    Randomly sample n_per_class images per class from a Dataset, 
    plot them in a grid with prediction and confidence.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Group indices by label
    class_to_indices = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    # Sample images
    sampled_indices = []
    for label, indices in class_to_indices.items():
        sampled_indices.extend(random.sample(indices, min(n_per_class, len(indices))))

    # Prepare grid
    n_images = len(sampled_indices)
    cols = min(5, n_images)
    rows = math.ceil(n_images / cols)
    plt.figure(figsize=(cols*3, rows*3))

    for i, idx in enumerate(sampled_indices):
        image, label = dataset[idx]
        image_tensor = image.unsqueeze(0)

        # Predict using helper function
        pred_idx, conf = _predict_image(model, image_tensor, device)
        predicted_label = class_names[pred_idx]
        confidence_percent = conf * 100

         
        image_denorm = denormalize(image.clone(), mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

        # Figure size in inches
        fig_width, fig_height = plt.gcf().get_size_inches()

        # Subplot width/height
        subplot_width = fig_width / cols
        subplot_height = fig_height / rows

        # Scale font based on the smaller dimension of the subplot
        font_size = min(subplot_width, subplot_height) * 2  # tweak multiplier
        # Plot
        plt.subplot(rows, cols, i+1)
        plt.imshow(transforms.ToPILImage()(image_denorm))
        plt.title(f"{predicted_label}\n({confidence_percent:.1f}%)", fontsize = font_size, pad = 4)
        plt.axis('off')

    plt.tight_layout(pad = 2)
    plt.show()

