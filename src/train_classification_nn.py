import os, argparse
import torch, torchvision
import networks


parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

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
    type=str, required=True, help='Path to save checkpoint file ')

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
    #dataset_train = None
    if args.dataset == 'cifar10':
        dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
    elif args.dataset == 'mnist':
        dataset_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms_train)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # TODO: Setup a dataloader (iterator) to fetch from the training set using
    # torch.utils.data.DataLoader and set shuffle=True, drop_last=True, num_workers=2
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.n_batch, shuffle=True, drop_last=True, num_workers=2)

    # TODO: Define the possible classes in depending on args.dataset variable
    class_names = dataset_train.classes if hasattr(dataset_train, 'classes') else None

    # TODO: Get number of classes in dataset
    n_class = len(class_names) if class_names is not None else None

    '''
    Set up model and optimizer
    '''
    # TODO: Compute number of input features depending on args.dataset variable
    #n_input_feature = None
    if args.dataset == 'cifar10':
        n_input_feature = 3 * 32 * 32  # 3 channels, 32x32 image size for CIFAR-10
    elif args.dataset == 'mnist':
        n_input_feature = 28 * 28  # 1 channel, 28x28 image size for MNIST
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # TODO: Instantiate network
    net = networks.NeuralNetwork(n_input_feature, n_class)

    # TODO: Setup learning rate SGD optimizer and step function scheduler
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_period, gamma=args.learning_rate_decay)
    '''
    Train network and store weights
    '''
    # TODO: Set network to training mode
    net.train()

    # TODO: Train network
    # net = None


    for epoch in range(args.n_epoch):
        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs.view(inputs.size(0), -1))

            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

        # Update scheduler (learning rate decay)
        scheduler.step()

        # Print training progress (optional)
        print(f"Epoch [{epoch+1}/{args.n_epoch}], Loss: {loss.item():.4f}")

    # TODO: Save weights into checkpoint
    torch.save({
        'epoch': args.n_epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, args.checkpoint_path)