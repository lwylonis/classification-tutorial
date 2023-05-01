import os, argparse
import torch, torchvision


parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoint file')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()


if __name__ == '__main__':

    '''
    Set up dataloading
    '''
    # TODO: Create transformations to apply to data during testing
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    transforms_test = None

    # TODO: Construct testing dataset based on args.dataset variable
    dataset_test = None

    # TODO: Setup a dataloader (iterator) to fetch from the testing set using
    # torch.utils.data.DataLoader and set shuffle=False, drop_last=False, num_workers=2
    # Set batch_size to 25
    dataloader_test = None

    '''
    Set up model
    '''
    # TODO: Compute number of input features based on args.dataset variable
    n_input_feature = None

    # TODO: Instantiate network
    net = None

    '''
    Restore weights and evaluate network
    '''
    # TODO: Load network from checkpoint
    checkpoint = None

    # TODO: Set network to evaluation mode

    # TODO: Evaluate network on testing set
