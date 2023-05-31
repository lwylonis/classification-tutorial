import torch
import networks
from matplotlib import pyplot as plt


class ClassificationModel(object):
    '''
    Classification model class that supports VGG11 and ResNet18 encoders

    Arg(s):
        encoder_type : str
            encoder options to build: vggnet11, resnet18, etc.
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 encoder_type,
                 device=torch.device('cuda')):

        self.device = device

        # TODO: Instantiate VGG11 and ResNet18 encoders and decoders based on
        # https://arxiv.org/pdf/1409.1556.pdf
        # https://arxiv.org/pdf/1512.03385.pdf
        # Decoder should use
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

        if encoder_type == 'vggnet11':
            self.encoder = None
            self.decoder = torch.nn.Sequential(None)
        elif encoder_type == 'resnet18':
            self.encoder = None
            self.decoder = torch.nn.Sequential(None)
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        # TODO: Move encoder and decoder to device

    def transform_input(self, images):
        '''
        Transforms input based on model arguments and settings

        Arg(s):
            images : torch.Tensor[float32]
                N x C x H x W images
        Returns:
            torch.Tensor[float32] : transformed N x C x H x W images
        '''

        # TODO: Perform normalization based on
        # https://arxiv.org/pdf/1409.1556.pdf
        # https://arxiv.org/pdf/1512.03385.pdf

        if self.encoder_type == 'vggnet11':
            pass
        elif self.encoder_type == 'resnet18':
            pass

        return None

    def forward(self, image):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
        Returns:
            torch.Tensor[float32] : N x K predicted class confidences
        '''

        # TODO: Implement forward function

        return None

    def compute_loss(self, output, label):
        '''
        Compute cross entropy loss

        Arg(s):
            output : torch.Tensor[float32]
                N x K predicted class confidences
            label : torch.Tensor[int]
                ground truth class labels
        Returns:
            float : loss averaged over the batch
            dict[str, float] : dictionary of loss related tensors
        '''

        # TODO: Compute cross entropy loss
        loss = None

        loss_info = {
            'loss' : loss
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return None

    def train(self):
        '''
        Sets model to training mode
        '''

        pass

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        pass

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device

        # TODO: Move encoder and decoder to device

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # TODO: Wrap encoder and decoder in
        # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                lists of paths to model weights
            optimizer : torch.optim or None
                current optimizer
        Returns:
            int : training step
            torch.optim : restored optimizer or None if no optimizer is passed in
        '''

        # TODO: Restore the weights from checkpoint
        # Encoder and decoder are keyed using 'encoder_state_dict' and 'decoder_state_dict'
        # If optimizer is given, then save its parameters using key 'optimizer_state_dict'
        pass

    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                list path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        # TODO: Save the weights into checkpoint
        # Encoder and decoder are keyed using 'encoder_state_dict' and 'decoder_state_dict'
        # If optimizer is given, then save its parameters using key 'optimizer_state_dict'
        pass

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image,
                    output,
                    ground_truth,
                    scalars={},
                    n_image_per_summary=16):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image : torch.Tensor[float32] 640 x 480
                image at time step
            output : torch.Tensor[float32]
                N
            label : torch.Tensor[float32]
                ground truth force measurements or ground truth bounding box and force measurements
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display
        '''

        with torch.no_grad():

            image_summary = image[0:n_image_per_summary, ...]

            # TODO: Move image_summary to CPU using cpu()

            # TODO: Convert image_summary to numpy using numpy() and swap dimensions from NCHW to NHWC
            n_batch, n_channel, n_height, n_width = image_summary.shape

            # TODO: Create plot figure of size n x n using log_utils

            # TODO: Log image summary to Tensorboard with <tag>_image as its summary tag name using
            # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure

            plt.tight_layout()

            plt.cla()
            plt.clf()
            plt.close()

            # TODO: Log scalars to Tensorboard with <tag>_<name> as its summary tag name using
            # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar
            for (name, value) in scalars.items():
                pass
