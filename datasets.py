from typing import Tuple
from torchvision import datasets, transforms


class TransformedDataset(datasets.CIFAR10):
    """
    Custom dataset class extending CIFAR10 dataset with additional transformation capabilities.

    Args:
        root (str, optional): Root directory where the dataset is stored. Default is "./data".
        train (bool, optional): Specifies if the dataset is for training or testing. Default is True.
        download (bool, optional): If True, downloads the dataset from the internet and places it in the root directory. Default is True.
        transform (list, optional): List of transformations to apply to the images. Default is None.

    """
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, transform: list = None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple:
        """
        Retrieves the item at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple: A tuple containing the transformed image and its label.

        """
        image, label = self.data[index], self.targets[index]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label
