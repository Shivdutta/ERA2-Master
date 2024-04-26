import math
from typing import NoReturn

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataclasses import dataclass
from typing import NoReturn
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix


# ---------------------------- DATA SAMPLES ----------------------------
def display_mnist_data_samples(dataset: 'DataLoader object', number_of_samples: int) -> NoReturn:
    """
    Function to display samples for dataloader
    :param dataset: Train or Test dataset transformed to Tensor
    :param number_of_samples: Number of samples to be displayed
    """
    # Get batch from the data_set
    batch_data = []
    batch_label = []
    for count, item in enumerate(dataset):
        if not count <= number_of_samples:
            break
        batch_data.append(item[0])
        batch_label.append(item[1])

    # Plot the samples from the batch
    fig = plt.figure()
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Plot the samples from the batch
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(), cmap='gray')
        plt.title(batch_label[i])
        plt.xticks([])
        plt.yticks([])


def display_cifar_data_samples(data_set, number_of_samples: int, classes: list):
    """
    Function to display samples for data_set
    :param data_set: Train or Test data_set transformed to Tensor
    :param number_of_samples: Number of samples to be displayed
    :param classes: Name of classes to be displayed
    """
    # Get batch from the data_set
    batch_data = []
    batch_label = []
    for count, item in enumerate(data_set):
        if not count <= number_of_samples:
            break
        batch_data.append(item[0])
        batch_label.append(item[1])
    batch_data = torch.stack(batch_data, dim=0).numpy()

    # Plot the samples from the batch
    fig = plt.figure()
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        plt.tight_layout()
        plt.imshow(np.transpose(batch_data[i].squeeze(), (1, 2, 0)))
        plt.title(classes[batch_label[i]])
        plt.xticks([])
        plt.yticks([])


# ---------------------------- MISCLASSIFIED DATA ----------------------------
def display_cifar_misclassified_data(data: list,
                                     classes: list[str],
                                     inv_normalize: transforms.Normalize,
                                     number_of_samples: int = 10):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(10, 10))

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = data[i][0].squeeze().to('cpu')
        img = inv_normalize(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])


def display_mnist_misclassified_data(data: list,
                                     number_of_samples: int = 10):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(8, 5))

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = data[i][0].squeeze(0).to('cpu')
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(r"Correct: " + str(data[i][1].item()) + '\n' + 'Output: ' + str(data[i][2].item()))
        plt.xticks([])
        plt.yticks([])


# ---------------------------- AUGMENTATION SAMPLES ----------------------------
def visualize_cifar_augmentation(data_set, data_transforms):
    """
    Function to visualize the augmented data
    :param data_set: Dataset without transformations
    :param data_transforms: Dictionary of transforms
    """
    sample, label = data_set[6]
    total_augmentations = len(data_transforms)

    fig = plt.figure(figsize=(10, 5))
    for count, (key, trans) in enumerate(data_transforms.items()):
        if count == total_augmentations - 1:
            break
        plt.subplot(math.ceil(total_augmentations / 5), 5, count + 1)
        augmented = trans(image=sample)['image']
        plt.imshow(augmented)
        plt.title(key)
        plt.xticks([])
        plt.yticks([])


def visualize_mnist_augmentation(data_set, data_transforms):
    """
    Function to visualize the augmented data
    :param data_set: Dataset to visualize the augmentations
    :param data_transforms: Dictionary of transforms
    """
    sample, label = data_set[6]
    total_augmentations = len(data_transforms)

    fig = plt.figure(figsize=(10, 5))
    for count, (key, trans) in enumerate(data_transforms.items()):
        if count == total_augmentations - 1:
            break
        plt.subplot(math.ceil(total_augmentations / 5), 5, count + 1)
        img = trans(sample).to('cpu')
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(key)
        plt.xticks([])
        plt.yticks([])


# ---------------------------- LOSS AND ACCURACIES ----------------------------
def display_loss_and_accuracies(train_losses: list,
                                train_acc: list,
                                test_losses: list,
                                test_acc: list,
                                plot_size: tuple = (10, 10)) -> NoReturn:
    """
    Function to display training and test information(losses and accuracies)
    :param train_losses: List containing training loss of each epoch
    :param train_acc: List containing training accuracy of each epoch
    :param test_losses: List containing test loss of each epoch
    :param test_acc: List containing test accuracy of each epoch
    :param plot_size: Size of the plot
    """
    # Create a plot of 2x2 of size
    fig, axs = plt.subplots(2, 2, figsize=plot_size)

    # Plot the training loss and accuracy for each epoch
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    # Plot the test loss and accuracy for each epoch
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


# ---------------------------- Feature Maps and Kernels ----------------------------

@dataclass
class ConvLayerInfo:
    """
    Data Class to store Conv layer's information
    """
    layer_number: int
    weights: torch.nn.parameter.Parameter
    layer_info: torch.nn.modules.conv.Conv2d


class FeatureMapVisualizer:
    """
    Class to visualize Feature Map of the Layers
    """

    def __init__(self, model):
        """
        Contructor
        :param model: Model Architecture
        """
        self.conv_layers = []
        self.outputs = []
        self.layerwise_kernels = None

        # Disect the model
        counter = 0
        model_children = model.children()
        for children in model_children:
            if type(children) == nn.Sequential:
                for child in children:
                    if type(child) == nn.Conv2d:
                        counter += 1
                        self.conv_layers.append(ConvLayerInfo(layer_number=counter,
                                                              weights=child.weight,
                                                              layer_info=child)
                                                )

    def get_model_weights(self):
        """
        Method to get the model weights
        """
        model_weights = [layer.weights for layer in self.conv_layers]
        return model_weights

    def get_conv_layers(self):
        """
        Get the convolution layers
        """
        conv_layers = [layer.layer_info for layer in self.conv_layers]
        return conv_layers

    def get_total_conv_layers(self) -> int:
        """
        Get total number of convolution layers
        """
        out = self.get_conv_layers()
        return len(out)

    def feature_maps_of_all_kernels(self, image: torch.Tensor) -> dict:
        """
        Get feature maps from all the kernels of all the layers
        :param image: Image to be passed to the network
        """
        image = image.unsqueeze(0)
        image = image.to('cpu')

        outputs = {}

        layers = self.get_conv_layers()
        for index, layer in enumerate(layers):
            image = layer(image)
            outputs[str(layer)] = image
        self.outputs = outputs
        return outputs

    def visualize_feature_map_of_kernel(self, image: torch.Tensor, kernel_number: int) -> None:
        """
        Function to visualize feature map of kernel number from each layer
        :param image: Image passed to the network
        :param kernel_number: Number of kernel in each layer (Should be less than or equal to the minimum number of kernel in the network)
        """
        # List to store processed feature maps
        processed = []

        # Get feature maps from all kernels of all the conv layers
        outputs = self.feature_maps_of_all_kernels(image)

        # Extract the n_th kernel's output from each layer and convert it to grayscale
        for feature_map in outputs.values():
            try:
                feature_map = feature_map[0][kernel_number]
            except IndexError:
                print("Filter number should be less than the minimum number of channels in a network")
                break
            finally:
                gray_scale = feature_map / feature_map.shape[0]
                processed.append(gray_scale.data.numpy())

        # Plot the Feature maps with layer and kernel number
        x_range = len(outputs) // 5 + 4
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(processed)):
            a = fig.add_subplot(x_range, 5, i + 1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            title = f"{list(outputs.keys())[i].split('(')[0]}_l{i + 1}_k{kernel_number}"
            a.set_title(title, fontsize=10)
        return fig

    def get_max_kernel_number(self):
        """
        Function to get maximum number of kernels in the network (for a layer)
        """
        layers = self.get_conv_layers()
        channels = [layer.out_channels for layer in layers]
        self.layerwise_kernels = channels
        return max(channels)

    def visualize_kernels_from_layer(self, layer_number: int):
        """
        Visualize Kernels from a layer
        :param layer_number: Number of layer from which kernels are to be visualized
        """
        # Get the kernels number for each layer
        self.get_max_kernel_number()

        # Zero Indexing
        layer_number = layer_number - 1
        _kernels = self.layerwise_kernels[layer_number]

        grid = math.ceil(math.sqrt(_kernels))

        fig = plt.figure(figsize=(5, 4))
        model_weights = self.get_model_weights()
        _layer_weights = model_weights[layer_number].cpu()
        for i, filter in enumerate(_layer_weights):
            plt.subplot(grid, grid, i + 1)
            plt.imshow(filter[0, :, :].detach(), cmap='gray')
            plt.axis('off')
        # plt.show()
        return fig


# ---------------------------- Confusion Matrix ----------------------------
def visualize_confusion_matrix(classes: list[str], device: str, model: 'DL Model',
                               test_loader: torch.utils.data.DataLoader):
    """
    Function to generate and visualize confusion matrix
    :param classes: List of class names
    :param device: cuda/cpu
    :param model: Model Architecture
    :param test_loader: DataLoader for test set
    """
    nb_classes = len(classes)
    device = 'cuda'
    cm = torch.zeros(nb_classes, nb_classes)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)

            preds = model(inputs)
            preds = preds.argmax(dim=1)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t, p] = cm[t, p] + 1

    # Build confusion matrix
    labels = labels.to('cpu')
    preds = preds.to('cpu')
    cf_matrix = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)

def get_summary(model: 'object of model architecture', input_size: tuple) -> NoReturn:
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size: Input data shape (Channels, Height, Width)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)


def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data


# -------------------- DATA STATISTICS --------------------
def get_mnist_statistics(data_set, data_set_type='Train'):
    """
    Function to return the statistics of the training data
    :param data_set: Training dataset
    :param data_set_type: Type of dataset [Train/Test/Val]
    """
    # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
    train_data = data_set.train_data
    train_data = data_set.transform(train_data.numpy())

    print(f'[{data_set_type}]')
    print(' - Numpy Shape:', data_set.train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', data_set.train_data.size())
    print(' - min:', torch.min(train_data))
    print(' - max:', torch.max(train_data))
    print(' - mean:', torch.mean(train_data))
    print(' - std:', torch.std(train_data))
    print(' - var:', torch.var(train_data))

    dataiter = next(iter(data_set))
    images, labels = dataiter[0], dataiter[1]

    print(images.shape)
    print(labels)

    # Let's visualize some of the images
    plt.imshow(images[0].numpy().squeeze(), cmap='gray')


def get_cifar_property(images, operation):
    """
    Get the property on each channel of the CIFAR
    :param images: Get the property value on the images
    :param operation: Mean, std, Variance, etc
    """
    param_r = eval('images[:, 0, :, :].' + operation + '()')
    param_g = eval('images[:, 1, :, :].' + operation + '()')
    param_b = eval('images[:, 2, :, :].' + operation + '()')
    return param_r, param_g, param_b


def get_cifar_statistics(data_set, data_set_type='Train'):
    """
    Function to get the statistical information of the CIFAR dataset
    :param data_set: Training set of CIFAR
    :param data_set_type: Training or Test data
    """
    # Images in the dataset
    images = [item[0] for item in data_set]
    images = torch.stack(images, dim=0).numpy()

    # Calculate mean over each channel
    mean_r, mean_g, mean_b = get_cifar_property(images, 'mean')

    # Calculate Standard deviation over each channel
    std_r, std_g, std_b = get_cifar_property(images, 'std')

    # Calculate min value over each channel
    min_r, min_g, min_b = get_cifar_property(images, 'min')

    # Calculate max value over each channel
    max_r, max_g, max_b = get_cifar_property(images, 'max')

    # Calculate variance value over each channel
    var_r, var_g, var_b = get_cifar_property(images, 'var')

    print(f'[{data_set_type}]')
    print(f' - Total {data_set_type} Images: {len(data_set)}')
    print(f' - Tensor Shape: {images[0].shape}')
    print(f' - min: {min_r, min_g, min_b}')
    print(f' - max: {max_r, max_g, max_b}')
    print(f' - mean: {mean_r, mean_g, mean_b}')
    print(f' - std: {std_r, std_g, std_b}')
    print(f' - var: {var_r, var_g, var_b}')

    # Let's visualize some of the images
    plt.imshow(np.transpose(images[1].squeeze(), (1, 2, 0)))


# -------------------- GradCam --------------------
def display_gradcam_output(data: list,
                           classes: list[str],
                           inv_normalize: transforms.Normalize,
                           model: 'DL Model',
                           target_layers: list['model_layer'],
                           targets=None,
                           number_of_samples: int = 10,
                           transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])
