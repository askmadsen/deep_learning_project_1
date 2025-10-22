from cat_dog_dataset import CatDogDataset
from custom_transformations import CustomTransformation
from torchvision import transforms

MODEL_DIR = "models/"
TRAINING_METRICS_DIR = "metrics/data/"
TRAINING_METRIC_IMAGES_DIR = "metrics/images/"

additional_train_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ]
transformer = CustomTransformation(image_size=(128, 128), other_transformations=[])

TRAIN_DATA = CatDogDataset(dogs_dir="catdog_data/train/dogs", cats_dir="catdog_data/train/cats", transform=transformer.get_train_transforms())
VALIDATION_DATA = CatDogDataset(dogs_dir="catdog_data/validation/dogs", cats_dir="catdog_data/validation/cats", transform=transformer.get_test_transforms())
TEST_DATA = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformer.get_test_transforms())
