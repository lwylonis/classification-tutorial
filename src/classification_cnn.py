import torch, torchvision, os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import log_utils


def train(model,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period,
          checkpoint_path,
          device):
    '''
    Trains the network using a learning rate scheduler

    Arg(s):
        model : ClassificationModel
            instance of the classification model
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch
        checkpoint_path : str
            path to save weights and Tensorboard summary
        device : str
            device to run on
    Returns:
        torch.nn.Module : trained network
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_checkpoint_path = os.path.join(checkpoint_path, 'model-{}.pth')
    event_path = os.path.join(checkpoint_path, 'events')

    train_summary_writer = SummaryWriter(event_path + '-train')

    # TODO: Move model to device using 'to(...)' function
    model = None

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:

            pass

        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Move images and labels to device
            images = None
            labels = None

            # TODO: Forward through the network
            outputs = None

            # TODO: Clear gradients so we don't accumlate them from previous batches

            # TODO: Compute loss function
            loss, loss_info = None, None

            # TODO: Update parameters by backpropagation

            # TODO: Accumulate total loss for the epoch
            total_loss = None

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch={}/{}  Loss: {:.3f}'.format(epoch + 1, n_epoch, mean_loss))

        # TODO: Save checkpoint after each epoch by using string format to insert epoch number to filename

    return model

def evaluate(model, dataloader, class_names, output_path, device):
    '''
    Evaluates the network on a dataset

    Arg(s):
        model : ClassificationModel
            instance of the classification model
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        class_names : list[str]
            list of class names to be used in plot
        output_path: str
            path to save output visualization
        device : str
            device to run on
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    # TODO: Move model to device
    model = None

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # TODO: Move images and labels to device
            images = None
            labels = None

            # TODO: Forward through the network
            outputs = None

            # TODO: Take the argmax over the outputs
            outputs = None

            # Accumulate number of samples
            n_sample = None

            # TODO: Check if our prediction is correct
            n_correct = None

    # TODO: Compute mean accuracy
    mean_accuracy = None

    print('Mean accuracy over {} images: {:.3f}%'.format(n_sample, mean_accuracy))

    # TODO: Convert the last batch of images back to original shape
    images = None

    # TODO: Move images back to cpu and to numpy array
    images = None

    # TODO: torch.Tensor operate in (N x C x H x W), convert it to (N x H x W x C)
    images = None

    # TODO: Move the last batch of labels to cpu and convert them to numpy and
    # map them to their corresponding class labels
    labels = None

    # TODO: Move the last batch of outputs to cpu, convert them to numpy and
    # map them to their corresponding class labels
    outputs = None

    # Convert images, outputs and labels to a lists of lists
    grid_size = 5

    images_display = []
    subplot_titles = []

    for row_idx in range(grid_size):
        # TODO: Get start and end indices of a row
        idx_start = None
        idx_end = None

        # TODO: Append images from start to end to image display array

        # TODO: Append text of 'output={}\nlabel={}' substituted with output and label to subplot titles

    # TODO: Plot images with class names and corresponding groundtruth label in a 5 by 5 grid
