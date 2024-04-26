import os
import math
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import CIFAR10

from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy

from datasets import TransformedDataset
from utils import get_cifar_statistics
from utils import visualize_cifar_augmentation, display_cifar_data_samples


class BasicBlock(LightningModule):
    """
    Basic building block for the ResNet architecture.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride value for the first convolutional layer. Defaults to 1.

    Attributes:
        expansion (int): Expansion factor for increasing dimensions in the block.

    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Initializes the BasicBlock.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride value for the first convolutional layer. Defaults to 1.

        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        Forward pass through the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LITResNet(LightningModule):
    """
    Lightning Module implementing the ResNet architecture for CIFAR-10 classification.

    Args:
        class_names (list): List of class names.
        data_dir (str, optional): Directory path to CIFAR-10 dataset. Defaults to '/data/'.

    Attributes:
        classes (list): List of class names.
        data_dir (str): Directory path to CIFAR-10 dataset.
        num_classes (int): Number of classes in the dataset.
        _learning_rate (float): Initial learning rate.
        inv_normalize (torchvision.transforms.Normalize): Inverse normalization transformation.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        accuracy (pl.metrics.Accuracy): Accuracy metric.
        train_transforms (torchvision.transforms.Compose): Transformations for training dataset.
        test_transforms (torchvision.transforms.Compose): Transformations for test dataset.
        stats_train (torchvision.datasets.CIFAR10): CIFAR-10 training dataset.
        stats_test (torchvision.datasets.CIFAR10): CIFAR-10 test dataset.
        cifar10_train (torch.utils.data.Subset): Subset of CIFAR-10 training dataset.
        cifar10_test (torch.utils.data.Subset): Subset of CIFAR-10 test dataset.
        cifar10_val (torch.utils.data.Subset): Subset of CIFAR-10 validation dataset.
        misclassified_data (list): List to store misclassified images.

    """
    def __init__(self, class_names, data_dir='/data/'):
        """
        Constructor
        """
        # Initialize the Module class
        super(LITResNet,self).__init__()

        # Initialize variables
        self.classes = class_names
        self.data_dir = data_dir
        self.num_classes = 10
        self._learning_rate = 0.03
        self.inv_normalize = transforms.Normalize(
            mean=[-0.50 / 0.23, -0.50 / 0.23, -0.50 / 0.23],
            std=[1 / 0.23, 1 / 0.23, 1 / 0.23]
        )
        self.batch_size = 512
        self.epochs = 24
        self.accuracy = Accuracy(task='multiclass',
                                 num_classes=10)
        self.train_transforms = transforms.Compose([transforms.ToTensor()])
        self.test_transforms = transforms.Compose([transforms.ToTensor()])
        self.stats_train = None
        self.stats_test = None
        self.cifar10_train = None
        self.cifar10_test = None
        self.cifar10_val = None
        self.misclassified_data = None

        # Defined Layers for the model
        self.prep_layer = None
        self.custom_block1 = None
        self.custom_block2 = None
        self.custom_block3 = None
        self.resnet_block1 = None
        self.resnet_block3 = None
        self.pool4 = None
        self.fc = None
        self.dropout_value = None
        self.model_layers(BasicBlock, [2, 2, 2, 2])

    # ##################################################################################################
    # ################################ Model Architecture Related Hooks ################################
    # ##################################################################################################
    def model_layers(self, block, num_blocks, num_classes=10):
        """
        Method to initialize layers for the model
        """
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Model Prediction
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    # ##################################################################################################
    # ############################## Training Configuration Related Hooks ##############################
    # ##################################################################################################

    def configure_optimizers(self):
        """
        Method to configure the optimizer and learning rate scheduler
        """
        learning_rate = 0.03
        weight_decay = 1e-4
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self._learning_rate,
                                                        steps_per_epoch=len(self.train_dataloader()),
                                                        epochs=self.epochs,
                                                        pct_start=5 / self.epochs,
                                                        div_factor=100,
                                                        three_phase=False,
                                                        final_div_factor=100,
                                                        anneal_strategy="linear"
                                                        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @property
    def learning_rate(self) -> float:
        """
        Method to get the learning rate value
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        """
        Method to set the learning rate value
        :param value: Updated value of learning rate
        """
        self._learning_rate = value

    def set_training_confi(self, *, epochs, batch_size):
        """
        Method to set parameters required for model training
        :param epochs: Number of epochs for which model is to be trained
        :param batch_size: Batch Size
        """
        self.epochs = epochs
        self.batch_size = batch_size

    # #################################################################################################
    # ################################## Training Loop Related Hooks ##################################
    # #################################################################################################
    def training_step(self, train_batch, batch_index):
        """
        Method called on training dataset to train the model
        :param train_batch: Batch containing images and labels
        :param batch_index: Index of the batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Method called on validation dataset to check if the model is learning
        :param batch: Batch containing images and labels
        :param batch_idx: Index of the batch
        """
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Method called on test dataset to check model performance on unseen data
        :param batch: Batch containing images and labels
        :param batch_idx: Index of the batch
        """
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    # ##############################################################################################
    # ##################################### Data Related Hooks #####################################
    # ##############################################################################################

    def set_transforms(self, train_set_transforms: dict, test_set_transforms: dict):
        """
        Method to set the transformations to be done on training and test datasets
        :param train_set_transforms: Dictionary of transformations for training dataset
        :param test_set_transforms: Dictionary of transformations for test dataset
        """
        self.train_transforms = A.Compose(train_set_transforms.values())
        self.test_transforms = A.Compose(test_set_transforms.values())

    def prepare_data(self):
        """
        Method to download the dataset
        """
        self.stats_train = CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        self.stats_test = CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage=None):
        """
        Method to create Split the dataset into train, test and val
        """
        # Only if dataset is not already split, perform the split operation
        if not self.cifar10_train and not self.cifar10_test and not self.cifar10_val:

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                cifar10_full = TransformedDataset(self.data_dir, train=True, download=True, transform=self.train_transforms)
                self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45_000, 5_000])

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.cifar10_test = TransformedDataset(self.data_dir, train=False, download=True,
                                                 transform=self.test_transforms)

    def train_dataloader(self):
        """
        Method to return the DataLoader for Training set
        """
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=os.cpu_count())

    def val_dataloader(self):
        """
        Method to return the DataLoader for the Validation set
        """
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        """
        Method to return the DataLoader for the Test set
        """
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=os.cpu_count())

    def get_statistics(self, data_set_type="Train"):
        """
        Method to get the statistics for CIFAR10 dataset
        """
        # Execute self.prepare_data() only if not done earlier
        if not self.stats_train and not self.stats_test:
            self.prepare_data()

        # Print stats for selected dataset
        if data_set_type == "Train":
            get_cifar_statistics(self.stats_train)
        else:
            get_cifar_statistics(self.stats_test, data_set_type="Test")

    def display_data_samples(self, dataset="train", num_of_images=20):
        """
        Method to display data samples
        """
        # Execute self.prepare_data() only if not done earlier
        try:
            assert self.stats_train
        except AttributeError:
            self.prepare_data()

        if dataset == "train":
            display_cifar_data_samples(self.stats_train, num_of_images, self.classes)
        else:
            display_cifar_data_samples(self.stats_test, num_of_images, self.classes)

    @staticmethod
    def visualize_augmentation(aug_set_transforms: dict):
        """
        Method to visualize augmentations
        :param aug_set_transforms: Dictionary of transformations to be visualized
        """
        aug_train = TransformedDataset('./data', train=True, download=True)
        visualize_cifar_augmentation(aug_train, aug_set_transforms)

    # #############################################################################################
    # ############################## Misclassified Data Related Hooks ##############################
    # #############################################################################################

    def get_misclassified_data(self):
        """
        Function to run the model on test set and return misclassified images
        """
        if self.misclassified_data:
            return self.misclassified_data

        self.misclassified_data = []
        self.prepare_data()
        self.setup()

        test_loader = self.test_dataloader()

        # Reset the gradients
        with torch.no_grad():
            # Extract images, labels in a batch
            for data, target in test_loader:

                # Migrate the data to the device
                data, target = data.to(self.device), target.to(self.device)

                # Extract single image, label from the batch
                for image, label in zip(data, target):

                    # Add batch dimension to the image
                    image = image.unsqueeze(0)

                    # Get the model prediction on the image
                    output = self.forward(image)

                    # Convert the output from one-hot encoding to a value
                    pred = output.argmax(dim=1, keepdim=True)

                    # If prediction is incorrect, append the data
                    if pred != label:
                        self.misclassified_data.append((image, label, pred))
        return self.misclassified_data

    def display_data_samples(self, dataset="train", num_of_images=20):
        """
        Method to display data samples
        """
        # Execute self.prepare_data() only if not done earlier
        try:
            assert self.stats_train
        except AttributeError:
            self.prepare_data()
    
        if dataset == "train":
            display_cifar_data_samples(self.stats_train, num_of_images, self.classes)
        else:
            display_cifar_data_samples(self.stats_test, num_of_images, self.classes)
            
    def display_cifar_misclassified_data(self, number_of_samples: int = 10):
        """
        Function to plot images with labels
        :param number_of_samples: Number of images to print
        """
        if not self.misclassified_data:
            self.misclassified_data = self.get_misclassified_data()

        fig = plt.figure(figsize=(10, 10))

        x_count = 5
        y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

        for i in range(number_of_samples):
            plt.subplot(y_count, x_count, i + 1)
            img = self.misclassified_data[i][0].squeeze().to('cpu')
            img = self.inv_normalize(img)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.title(
                r"Correct: " + self.classes[self.misclassified_data[i][1].item()] + '\n' + 'Output: ' + self.classes[
                    self.misclassified_data[i][2].item()])
            plt.xticks([])
            plt.yticks([])
