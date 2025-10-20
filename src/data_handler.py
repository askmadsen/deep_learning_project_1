import json
import matplotlib.pyplot as plt
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