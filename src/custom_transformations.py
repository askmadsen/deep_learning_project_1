from torchvision import transforms

class CustomTransformation:
    def __init__(self, image_size: tuple[int,int], other_transformations: list) -> None:
        self.image_size = image_size
        self.other_transformations = other_transformations
    
    def get_train_transforms(self):
        transformations = self.other_transformations
        #transformations.insert(0, transforms.Grayscale(num_output_channels=1))
        transformations.insert(0, transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)))
        transformations.append(transforms.ToTensor())
        transformations.append(transforms.Normalize((0.5,), (0.5,)))
        transformations.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))
        return transforms.Compose(transformations)
    
    def get_test_transforms(self):
        transformations = [transforms.Resize(self.image_size), transforms.ToTensor()]
        transformations.append(transforms.Normalize((0.5,), (0.5,)))
        return transforms.Compose(transformations)
    
    # Add this at the start of transform:     transforms.Grayscale(num_output_channels=3),
