from cat_dog_classifier import CatDogClassifier
from cat_dog_classifier_new import CatDogClassifierNew
from cat_dog_dataset import CatDogDataset
from cat_dog_classifier_good import CatDogClassifierGood
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from data_handler import plot_metrics, plot_confusion_matrix, predict_grid_dataset, predict_single
import pandas as pd
import os
import json
from config import TEST_DATA as test_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID = "1406"
MODEL_PATH = f"models/good_{ID}.pth"



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

def summarize_seeds_with_test(model, json_base_path, model_base_path, seeds, test_loader, device, last_n=10):
    """
    Summarize training for multiple seeds and compute test accuracy,
    returning a pandas DataFrame.
    
    Args:
        json_base_path (str): Base path prefix for JSON metric files
        model_base_path (str): Base path prefix for model files
        seeds (list): List of integer seeds
        test_loader: PyTorch DataLoader for test set
        device: torch device
        last_n (int): Number of epochs to average for train/val acc
    
    Returns:
        pd.DataFrame: Each row corresponds to a seed with columns:
                      ['seed', 'train_acc_avg', 'train_acc_max',
                       'val_acc_avg', 'val_acc_max', 'test_acc']
    """
    rows = []

    for seed in seeds:
        json_path = f"{json_base_path}good_{seed}.json"
        model_path = f"{model_base_path}good_{seed}.pth"

        if not os.path.exists(json_path):
            print(f"Warning: JSON file {json_path} not found, skipping seed {seed}")
            continue
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found, skipping seed {seed}")
            continue

        # Load JSON metrics
        with open(json_path, "r") as f:
            data = json.load(f)

        train_acc_avg = sum(data["train_acc"][-last_n:]) / last_n
        train_acc_max = max(data["train_acc"])
        val_acc_avg = sum(data["validation_acc"][-last_n:]) / last_n
        val_acc_max = max(data["validation_acc"])

        # Load model and compute test accuracy
        classifier =  model  # replace with your model class
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        classifier.to(device)
        classifier.eval()

        criterion = nn.CrossEntropyLoss()
        acc, loss = test_model(classifier, test_loader, criterion, device)

        # Append row
        rows.append({
            "seed": seed,
            "train_acc_avg": round(train_acc_avg, 3),
            "train_acc_max": round(train_acc_max, 3),
            "val_acc_avg": round(val_acc_avg, 3),
            "val_acc_max": round(val_acc_max, 3),
            "test_acc": round(acc, 3)
        })

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df



if __name__ == "__main__":
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    #seeds = [42, 1054, 1406, 3772, 4609, 5168, 6525, 7214, 8263, 9210]
    seeds = [6525, 4609, 1406]
    model = CatDogClassifierNew()


    tasks = {
        "plot_metrics": lambda: plot_metrics("metrics/plot/good_1406.png", "metrics/data/good_1406.json"),
        "summarize_seeds": lambda: print(summarize_seeds_with_test(model, "metrics/data/", "models/", seeds, test_loader, DEVICE)),
        "confusion_matrix": lambda: plot_confusion_matrix(
            model=model,
            model_path=MODEL_PATH,
            device=DEVICE,
            test_loader=test_loader,
            class_names=["Dog", "Cat"]
        ),
        "predict_single": lambda: predict_single(
            model=model,
            model_path=MODEL_PATH,
            image_path="garfield.jpg",
            transform=test_dataset.transform,
            device=DEVICE
        ),
        "predict_grid": lambda: predict_grid_dataset(
            model=model,
            model_path=MODEL_PATH,
            dataset=test_dataset,
            device=DEVICE,
            n_per_class=10,
            class_names=["Dog", "Cat"]
        )
    }

    # Choose the task by key
    selected_task = "predict_grid"  
    tasks[selected_task]()



    

    