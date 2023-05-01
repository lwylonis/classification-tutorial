import torch


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
        self.fully_connected_layer1 = None
        self.fully_connected_layer2 = None
        self.fully_connected_layer3 = None
        self.fully_connected_layer4 = None
        self.fully_connected_layer5 = None

        # TODO: Define output layer
        self.output = None

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
        output_fc1 = None
        output_fc2 = None
        output_fc3 = None
        output_fc4 = None
        output_fc5 = None

        output_logits = None

        return output_logits
