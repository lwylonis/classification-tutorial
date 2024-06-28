import os, argparse
import torch, torchvision
import networks


parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

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

    if args.dataset == 'cifar10':
        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
    elif args.dataset == 'mnist':
        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms_test)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=25, shuffle=False, drop_last=False, num_workers=2)

    # TODO: Create transformations to apply to data during testing
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    ###transforms_test = None

    # TODO: Construct testing dataset based on args.dataset variable
    ###dataset_test = None

    # TODO: Setup a dataloader (iterator) to fetch from the testing set using
    # torch.utils.data.DataLoader and set shuffle=False, drop_last=False, num_workers=2
    # Set batch_size to 25
    ###dataloader_test = None

    '''
    Set up model
    '''
    # TODO: Compute number of input features based on args.dataset variable
    ###n_input_feature = None
    if args.dataset == 'cifar10':
        n_input_feature = 3 * 32 * 32  # 3 channels, 32x32 image size
    elif args.dataset == 'mnist':
        n_input_feature = 28 * 28  # 1 channel, 28x28 image size
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # TODO: Instantiate network
    net = networks.NeuralNetwork(n_input_feature, 10)

    '''
    Restore weights and evaluate network
    '''
    # TODO: Load network from checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    # TODO: Set network to evaluation mode
    net.eval()

    # TODO: Evaluate network on testing set
    device = torch.device(args.device)
    net.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images.view(images.size(0), -1))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on the testing set: {accuracy:.2%}')