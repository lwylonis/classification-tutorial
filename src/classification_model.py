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


        self.encoder_type = encoder_type

        if encoder_type == 'vggnet11':
            self.encoder = networks.VGGNet11Encoder()
            self.decoder = torch.nn.Sequential()
        elif encoder_type == 'resnet18':
            self.encoder = networks.ResNet18Encoder()
            self.decoder = torch.nn.Sequential()
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        # TODO: Move encoder and decoder to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

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

        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        images = (images - mean[None, :, None, None]) / std[None, :, None, None]

        # if self.encoder_type == 'vggnet11':
        #     pass
        # elif self.encoder_type == 'resnet18':
        #     pass

        return images

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
        features, _ = self.encoder(image)
        output = self.decoder(features)

        return output

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

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, label)

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

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''
        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)

        self.device = device

        # TODO: Move encoder and decoder to device

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # TODO: Wrap encoder and decoder in
        # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)

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

        checkpoint = torch.load(restore_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['step'], optimizer

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
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }, checkpoint_path)

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


            image_summary = image[0:n_image_per_summary, ...].cpu().numpy().transpose(0, 2, 3, 1)
            fig = plt.figure(figsize=(n_image_per_summary, 1))
            for idx in range(n_image_per_summary):
                ax = fig.add_subplot(1, n_image_per_summary, idx + 1)
                ax.imshow(image_summary[idx])
                ax.axis('off')

            summary_writer.add_figure(f'{tag}_image', fig, global_step=step)

            for name, value in scalars.items():
                summary_writer.add_scalar(f'{tag}_{name}', value, global_step=step)

            # image_summary = image[0:n_image_per_summary, ...]

            # # TODO: Move image_summary to CPU using cpu()

            # # TODO: Convert image_summary to numpy using numpy() and swap dimensions from NCHW to NHWC
            # n_batch, n_channel, n_height, n_width = image_summary.shape

            # # TODO: Create plot figure of size n x n using log_utils

            # # TODO: Log image summary to Tensorboard with <tag>_image as its summary tag name using
            # # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure

            # plt.tight_layout()

            # plt.cla()
            # plt.clf()
            # plt.close()

            # # TODO: Log scalars to Tensorboard with <tag>_<name> as its summary tag name using
            # # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar
            # for (name, value) in scalars.items():
            #     pass
