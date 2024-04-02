import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Session10Net(nn.Module):
    
    """
    David's Model Architecture for Session-10 CIFAR10 dataset
    """
    def __init__(self):
        """
        Constructor
        """
        # Initialize the Module class
        super(Session10Net, self).__init__()

        #Prepare data for Prep Layer Convolutional Block
        in_channels = 3
        out_channels_list = [64]
        kernel_size_list = [3]
        stride_list = [1]
        padding_list = [1]
        dilation_list = [0]
        conv_type = ['standard']
        max_pool_list=[0]
        self.prep_layer = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,
                            max_pool_list, activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,
                            last_layer=False)

        #self.prep_layer = self.standard_conv_layer(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)

        #Prepare data for Convolutional Block 1 
        in_channels = 64
        out_channels_list = [128]
        kernel_size_list = [3]
        stride_list = [1]
        padding_list = [1]
        dilation_list = [0]
        conv_type = ['standard']
        max_pool_list=[2]
        self.custom_block1 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,max_pool_list,
                        activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)
        self.resnet_block1 = self.resnet_block(channels=128)

        # # Convolutional Block-1
        # #self.custom_block1 = Session10Net.custom_block(input_channels=64, output_channels=128)
        # #self.resnet_block1 = Session10Net.resnet_block(channels=128)

        #Prepare data for Convolutional Block 2 
        in_channels = 128
        out_channels_list = [256]
        kernel_size_list = [3]
        stride_list = [1]
        padding_list = [1]
        dilation_list = [0]
        conv_type = ['standard']
        max_pool_list = [2]
        self.custom_block2 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,max_pool_list,
                        activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)
        #Max pool of 2D  -- I have to send list of [T,F,T] Fi if true then add max pool

        # Convolutional Block-2
        #self.custom_block2 = Session10Net.custom_block(input_channels=128, output_channels=256)
        
        #Prepare data for Convolutional Block 3 
        in_channels = 256
        out_channels_list = [512]
        kernel_size_list = [3]
        stride_list = [1]
        padding_list = [1]
        dilation_list = [0]
        conv_type = ['standard']
        max_pool_list = [2]
        self.custom_block3 = self.get_conv_block(conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,max_pool_list,
                        activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False)
        self.resnet_block3 = self.resnet_block(channels=512)
        #Max pool of 2D  -- I have to send list of [T,F,T] Fi if true then add max pool

        # # Convolutional Block-3
        # #self.custom_block3 = Session10Net.custom_block(input_channels=256, output_channels=512)
        # #self.resnet_block3 = Session10Net.resnet_block(channels=512)

        # MaxPool Layer
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=2)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Model Prediction
        """
        # Prep Layer
        x = self.prep_layer(x)

        # Convolutional Block-1
        x = self.custom_block1(x)
        r1 = self.resnet_block1(x)
        x = x + r1

        # Convolutional Block-2
        x = self.custom_block2(x)

        # Convolutional Block-3
        x = self.custom_block3(x)
        r2 = self.resnet_block3(x)
        x = x + r2

        # MaxPool Layer
        x = self.pool4(x)

        # Fully Connected Layer
        x = x.view(-1, 512)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


    @staticmethod
    def resnet_block(channels):
        """
        Method to create a RESNET block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
            
    def get_conv_block(self, conv_type, in_channels, out_channels_list, kernel_size_list, stride_list, padding_list,dilation_list,max_pool_list=None,
                       activation_fn = nn.ReLU(inplace=True),normalization='batch',number_of_groups =None,last_layer=False):
        """
        Create a convolutional block consisting of multiple convolutional layers followed by normalization and activation.

        Args:
            conv_type (list): List specifying the type of each convolutional layer. Options: "standard", "depthwise", "dilated".
            in_channels (int): Number of input channels.
            out_channels_list (list): List of integers specifying the number of output channels for each convolutional layer.
            kernel_size_list (list): List of integers specifying the kernel size for each convolutional layer.
            stride_list (list): List of integers specifying the stride for each convolutional layer.
            padding_list (list): List of integers specifying the padding for each convolutional layer.
            dilation_list (list): List of integers specifying the dilation for each dilated convolutional layer.
            activation_fn (torch.nn.Module, optional): Activation function to apply after each convolutional layer. Default is ReLU.
            normalization (str, optional): Type of normalization layer. Options: "batch", "layer", "group". Default is "batch".
            number_of_groups (int, optional): Number of groups for group normalization. Required only if normalization is "group".
            last_layer (bool, optional): Flag indicating whether this block is the final layer in the network. Default is False.

        Returns:
            torch.nn.Sequential: Convolutional block consisting of convolutional layers followed by normalization and activation.

        Raises:
            AssertionError: If lengths of the lists specifying parameters for convolutional layers do not match.
        """
        assert len(out_channels_list) == len(kernel_size_list) == len(stride_list) == len(padding_list), "Lengths of lists should match"
        layers = []
        #print("len(out_channels_list)",len(out_channels_list))
        for i in range(len(out_channels_list)):
            if i == 0:
                if conv_type[i] == "standard":
                    _conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_list[i], stride=stride_list[i], kernel_size=kernel_size_list[i], bias=False, padding=padding_list[i])
                elif conv_type[i] == "depthwise":
                    _conv_layer = self.depthwise_conv(in_channels=in_channels, out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i])
                elif conv_type[i] == "dilated":
                    _conv_layer = self.dilated_conv(in_channels=in_channels, out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i], dilation=dilation_list[i])
        
                layers.append(_conv_layer)
            else:
                if conv_type[i] == "standard":
                    _conv_layer = nn.Conv2d(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i], stride=stride_list[i], kernel_size=kernel_size_list[i], bias=False, padding=padding_list[i])
                elif conv_type[i] == "depthwise":
                    _conv_layer = self.depthwise_conv(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i])
                elif conv_type[i] == "dilated":
                    _conv_layer = self.dilated_conv(in_channels=out_channels_list[i-1], out_channels=out_channels_list[i], stride=stride_list[i], padding=padding_list[i], dilation=dilation_list[i])
        
                layers.append(_conv_layer)
                    

            if not last_layer:
                _norm_layer = self.get_normalization_layer(normalization,out_channels_list[i],number_of_groups)
                if int(max_pool_list[i])>0:
                        layers.append(nn.MaxPool2d(kernel_size=max_pool_list[i], stride=max_pool_list[i]))

                layers.append(_norm_layer)
                layers.append(activation_fn)
                
        conv_layers = nn.Sequential(*layers)
        return conv_layers
        

    @staticmethod
    def get_normalization_layer(normalization,out_channels,number_of_groups = None):
        """
        Create a normalization layer based on the specified normalization technique.

        Args:
            normalization (str): Type of normalization layer. Options: "layer", "group", or any other value (defaults to "batch").
            out_channels (int): Number of output channels.
            number_of_groups (int, optional): Number of groups for group normalization. Required only if normalization is "group".

        Returns:
            torch.nn.Module: Normalization layer based on the specified technique.
        """
        if normalization == "layer":
            _norm_layer = nn.GroupNorm(1, out_channels)
        elif normalization == "group":
            if not number_of_groups:
                raise ValueError("Value of group is not defined")
            _norm_layer = nn.GroupNorm(number_of_groups, out_channels)
        else:
            _norm_layer = nn.BatchNorm2d(out_channels)
        
        return _norm_layer
    

        
    @staticmethod
    def depthwise_conv(in_channels, out_channels, stride=1, padding=0):
        """
        Create a depthwise separable convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride of the convolution. Default is 1.
            padding (int, optional): Padding added to all four sides of the input. Default is 0.

        Returns:
            torch.nn.Sequential: Depthwise separable convolutional layer consisting of a depthwise convolution followed by a pointwise convolution.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels, kernel_size=3, bias=False, padding=padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1, bias=False, padding=0)
        )

    @staticmethod
    def dilated_conv(in_channels, out_channels, stride=1, padding=0, dilation=1):
        """
        Create a dilated convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride of the convolution. Default is 1.
            padding (int, optional): Padding added to all four sides of the input. Default is 0.
            dilation (int, optional): Spacing between kernel elements. Default is 1.

        Returns:
            torch.nn.Sequential: Dilated convolutional layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, bias=False,
                      padding=padding, dilation=dilation)
        )

    
def get_summary(model, input_size) :       
    """
    Generate a summary of the given model.

    Args:
        model (torch.nn.Module): The neural network model.
        input_size (tuple): The input size of the model, typically in the format (batch_size, channels, height, width).

    Returns:
        str: A summary of the model architecture including details such as layer types, output shape, and number of parameters.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    return summary(network, input_size=input_size)

    
#unit test
# model = Session9Net()
# input_tensor = torch.randn(1, 3, 224, 224)  
# output_tensor = model(input_tensor)
# print(output_tensor.shape)
