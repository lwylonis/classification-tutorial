import os, argparse
import torch, torchvision

from torch.utils.data import DataLoader
import classification_model, classification_cnn

parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

# Network settings
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type to build: vggnet11, resnet18')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoint file')
parser.add_argument('--output_path',
    type=str, required=True, help='Path to save output')

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
    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # TODO: Construct testing dataset based on args.dataset variable
    if args.dataset == 'cifar10':
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
        class_names = dataset_test.classes
    elif args.dataset == 'mnist':
        dataset_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms_test)
        class_names = list(range(10))  # MNIST has digits from 0 to 9
    else:
        raise ValueError('Unsupported dataset: {}'.format(args.dataset))

    # TODO: Setup a dataloader (iterator) to fetch from the testing set using
    # torch.utils.data.DataLoader and set shuffle=False, drop_last=False, num_workers=2
    # Set batch_size to 25
    dataloader_test = DataLoader(dataset_test, batch_size=25, shuffle=False, drop_last=False, num_workers=2)

    '''
    Set up model
    '''
    # TODO: Instantiate network
    model = classification_model.ClassificationModel(encoder_type=args.encoder_type, device=args.device)

    '''
    Restore weights and evaluate network
    '''
    # TODO: Load network from checkpoint
    #checkpoint = torch.load(args.checkpoint_path)
    model.restore_model(args.checkpoint_path)

    # TODO: Set network to evaluation mode
    model.eval()

    # TODO: Evaluate network on testing set
    classification_cnn.evaluate(model, dataloader_test, class_names, args.output_path, args.device)
