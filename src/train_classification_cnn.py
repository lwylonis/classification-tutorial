import os, argparse
import torch, torchvision


parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

# Network settings
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type to build: vggnet11, resnet18')

# Training settings
parser.add_argument('--n_epoch',
    type=int, required=True, help='Number of passes through the full training dataset')
parser.add_argument('--learning_rate',
    type=float, required=True, help='Step size to update parameters')
parser.add_argument('--learning_rate_decay',
    type=float, required=True, help='Scaling factor to decrease learning rate at the end of each decay period')
parser.add_argument('--learning_rate_period',
    type=float, required=True, help='Number of epochs before reducing/decaying learning rat')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path directory to save checkpoints and Tensorboard summaries')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()


if __name__ == '__main__':

    # Create transformations to apply to data during training
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    transforms_train = torchvision.transforms.Compose([
        # TODO: Include random brightness, contrast, saturation, flip
        # and other augmentations of your choice
        torchvision.transforms.ToTensor()
    ])

    # TODO: Construct training dataset based on args.dataset variable
    dataset_train = None

    # TODO: Setup a dataloader (iterator) to fetch from the training set using
    # torch.utils.data.DataLoader and set shuffle=True, drop_last=True, num_workers=2
    dataloader_train = None

    # TODO: Define the possible classes in depending on args.dataset variable
    class_names = None

    # TODO: Get number of classes in dataset
    n_class = None

    '''
    Set up model and optimizer
    '''
    # TODO: Compute number of input features depending on args.dataset variable
    n_input_feature = None

    # TODO: Instantiate network
    model = None

    # TODO: Setup learning rate SGD optimizer and step function scheduler
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = None

    '''
    Train network and store weights
    '''
    # TODO: Set network to training mode

    # TODO: Train network
