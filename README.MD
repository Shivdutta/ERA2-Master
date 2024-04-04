# Master Repository 
- The intent of creating repository is to reuse common functionality for defining the network, data augumentation and model training/validation. 
 
This repository contains following files:

- `custom_resnet.py`: Model Architecture by [David C](https://github.com/davidcpage)
- `dataset.py`: This script contains class to use with Albumentations (https://github.com/albumentations-team/albumentations) library with PyTorch dataset
- `training_utils`: This script contains `train`, `test` functions required for model training along with `get_lr` function to get current learning rate
- `utils.py`: This script  contains utility functions to visualize data samples, augmentations, misclassified images, dataset statistics and model summary. 

## Code Details

- `utils.py`:
    - This python script contains collection of utility functions related to data processing, visualization, and model evaluation in PyTorch. Here's a summary of what each function does:

        - get_mnist_statistics: Computes statistics (min, max, mean, std, var) of the MNIST dataset.
        - get_cifar_statistics: Computes statistics (min, max, mean, std, var) of the CIFAR dataset.
        - plot_accuracy_losses: Plots training and test losses along with training and test accuracies.
        - display_data_samples: Displays samples from a dataset along with their labels.
        - plot_data: Plots images with their correct and predicted labels.
        - display_loss_and_accuracies: Displays training and test losses along with accuracies.
        - get_misclassified_data: Retrieves misclassified images from a test set.
        - display_mnist_data_samples: Displays samples from the MNIST dataset.
        - display_cifar_data_samples: Displays samples from the CIFAR dataset.
        - display_cifar_misclassified_data: Displays misclassified images from the CIFAR dataset.
        - display_mnist_misclassified_data: Displays misclassified images from the MNIST dataset.
        - visualize_cifar_augmentation: Visualizes augmented data from the CIFAR dataset.
        - visualize_mnist_augmentation: Visualizes augmented data from the MNIST dataset.

- `training_utils`:
    - This Python script defines several functions related to training and evaluating a PyTorch model:

        - get_correct_predictions: Calculates the total number of correct predictions given model predictions and correct labels.
        - train: Function to train the model on the training dataset. It iterates over each batch, calculates the loss, performs backpropagation, and updates the  weights using the optimizer. It also tracks training loss and accuracy.
        - test: Function to evaluate the model on the test dataset. It calculates the loss and accuracy on the test dataset without updating the model parameters.
        - get_lr: Function to extract the current learning rate from the optimizer.

    - These functions collectively facilitate the training and evaluation of a PyTorch model, including tracking metrics such as loss and accuracy during training and evaluation phases. The tqdm library is used for displaying progress bars during training.

- `dataset.py`:
    - This Python script defines a custom dataset class TransformedDataset that extends the functionality of the CIFAR10 dataset from torchvision. Here's a summary:

- `custom_resnet.py`:

    - This Python script defines a neural network model called Session10Net designed for the CIFAR10 dataset. Here's a summary of its key components:

        - Model Architecture:

            - The model consists of several convolutional blocks and a fully connected layer.
            - It utilizes standard convolutional layers and ResNet blocks.
            - The convolutional blocks are defined based on various parameters such as the type of convolution, number of channels, kernel size, etc.
            - The ResNet blocks consist of two convolutional layers followed by batch normalization and ReLU activation.

        - Forward Method:

            - The forward method defines the forward pass of the model, specifying how input data flows through the network.
            - It passes the input through the preparation layer, followed by three convolutional blocks, each with a ResNet block, and finally through a max-pooling layer and a fully connected layer.

        - Helper Methods:

            - get_conv_block: This method constructs a convolutional block consisting of multiple convolutional layers followed by normalization and activation.
            - resnet_block: This method creates a ResNet block with two convolutional layers.
            - get_normalization_layer: This method generates normalization layers based on the specified technique (batch normalization, layer normalization, or group normalization).
            - depthwise_conv and dilated_conv: These methods create depthwise separable and dilated convolutional layers, respectively.

- Summary Generation: 
    - The get_summary function generates a summary of the model architecture, including details such as layer types, output shape, and the number of parameters. It utilizes the torchsummary library to produce the summary.