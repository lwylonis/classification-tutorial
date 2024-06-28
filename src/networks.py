import torch
import net_utils


class NeuralNetwork(torch.nn.Module):
    '''
    Neural network class of fully connected layers

    Arg(s):
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
    '''

    def __init__(self, n_input_feature, n_output):
        super(NeuralNetwork, self).__init__()

        # Create your 6-layer neural network using fully connected layers with ReLU activations
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

        # TODO: Instantiate 5 fully connected layers and choose number of neurons i.e. 512
        self.fully_connected_layer1 = torch.nn.Linear(n_input_feature, 512)
        self.fully_connected_layer2 = torch.nn.Linear(512, 512)
        self.fully_connected_layer3 = torch.nn.Linear(512, 512)
        self.fully_connected_layer4 = torch.nn.Linear(512, 512)
        self.fully_connected_layer5 = torch.nn.Linear(512, 512)

        # TODO: Define output layer
        self.output = torch.nn.Linear(512, n_output)

    def forward(self, x):
        '''
        Forward pass through the neural network

        Arg(s):
            x : torch.Tensor[float32]
                tensor of N x d
        Returns:
            torch.Tensor[float32]
                tensor of n_output predicted class
        '''

        # TODO: Implement forward function
        output_fc1 = torch.relu(self.fully_connected_layer1(x))
        output_fc2 = torch.relu(self.fully_connected_layer2(output_fc1))
        output_fc3 = torch.relu(self.fully_connected_layer3(output_fc2))
        output_fc4 = torch.relu(self.fully_connected_layer4(output_fc3))
        output_fc5 = torch.relu(self.fully_connected_layer5(output_fc4))

        output_logits = self.output(output_fc5)

        return output_logits

class ResNet18Encoder(torch.nn.Module):
    '''
    ResNet18 encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNet18Encoder, self).__init__()

        assert len(n_filters) == 5

        activation_func = net_utils.activation_func(activation_func)

        # TODO: Implement ResNet encoder using ResNetBlock from net_utils.py
        # Based on https://arxiv.org/pdf/1512.03385.pdf

    def forward(self, x):
        '''
        Forward input x through a ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of intermediate feature maps used for skip connections
        '''

        layers = [x]

        # TODO: Implement forward function

        # Return latent and intermediate features
        return layers[-1], layers[1:-1]

class VGGNet11Encoder(torch.nn.Module):
    '''
    VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(VGGNet11Encoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        # TODO: Implement VGGNet encoder using VGGNetBlock from net_utils.py
        # Based on https://arxiv.org/pdf/1409.1556.pdf

    def forward(self, x):
        '''
        Forward input x through a VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of intermediate feature maps used for skip connections
        '''

        layers = [x]

        # TODO: Implement forward function

        # Return latent and intermediate features
        return layers[-1], layers[1:-1]
