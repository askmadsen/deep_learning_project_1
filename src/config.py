from cat_dog_dataset import CatDogDataset
from custom_transformations import CustomTransformation
from torchvision import transforms

MODEL_DIR = "models/"
TRAINING_METRICS_DIR = "metrics/data/"
TRAINING_METRIC_IMAGES_DIR = "metrics/images/"

additional_train_transforms =  [
    transforms.RandomHorizontalFlip(),          # keep left-right flips
    transforms.RandomRotation(10),              # slightly smaller rotations
    transforms.ColorJitter(
        brightness=0.1,                         # reduce intensity
        contrast=0.1,
        saturation=0.1,
        hue=0.03
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),                 # smaller translations
        scale=(0.97, 1.03)                      # smaller scaling
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))  # mild blur
    ]
    
transformer = CustomTransformation(image_size=(128, 128), other_transformations=additional_train_transforms)

TRAIN_DATA = CatDogDataset(dogs_dir="catdog_data/train/dogs", cats_dir="catdog_data/train/cats", transform=transformer.get_train_transforms())
VALIDATION_DATA = CatDogDataset(dogs_dir="catdog_data/validation/dogs", cats_dir="catdog_data/validation/cats", transform=transformer.get_test_transforms())
TEST_DATA = CatDogDataset(dogs_dir="catdog_data/test/dogs", cats_dir="catdog_data/test/cats", transform=transformer.get_test_transforms())
