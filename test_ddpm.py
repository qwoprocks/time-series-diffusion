# Portions of code in this file were created with reference to:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Assignment 3

import argparse
import os
import warnings
policy = 'color,translation,cutout' # If your dataset is as small as ours (e.g.,

warnings.filterwarnings("ignore")

# Numpy & Scipy imports
import numpy as np

# Torch imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Local imports
import utils
from data_loader import get_data_loader
from models import DCGenerator, DCDiscriminator
from diffusion_model import Unet, p_losses, sample
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from vanilla_gan import create_image_grid
import imageio

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(U):
    """Prints model information for the UNet.
    """
    print("                    U                  ")
    print("---------------------------------------")
    print(U)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    U = Unet(dim=opts.image_size, channels=3, dim_mults=(1, 2, 4, 8), self_condition=True)

    print_models(U)

    if torch.cuda.is_available():
        U.cuda()
        print('Models moved to GPU.')

    return U


def sampling_loop(train_dataloader, opts):
    """Runs the sampling loop.
        * Loads the checkpoint
        * Denoises the image iteratively to generate samples
    """

    # Create UNet
    U = create_model(opts)

    U.load_state_dict(torch.load('diffusion.pth')) #Load diffusion model checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_samples = opts.num_samples

    # sample n images
    U.eval()
    samples = sample(U, opts.image_size, batch_size=opts.batch_size, channels=3)
    print(samples[-1].min(), samples[-1].max())

    # code to save results
    all_images = np.empty((len(samples[-1]), 3, opts.image_size, opts.image_size))
    for i in range(opts.batch_size):
        img = samples[-1][i]
        img = np.clip((img + 1.0) * 0.5, 0, 1)
        all_images[i] = img
        img = np.transpose(img, (1, 2, 0))
        img = np.uint8(255 * img)
        imageio.imwrite(f'diffusion_outputs/diffusion_output_{i}.png', img)

    all_images = np.uint8(255 * all_images)
    imageio.imwrite('diffusion_outputs/merged_outputs.png', create_image_grid(all_images, ncols=4))

    # 500 timesteps, so shape of (10, bs, channels, img_sz, img_sz)
    skipped_samples = samples[::50]
    skipped_samples[-1] = samples[-1]
    all_images = np.clip((np.array(skipped_samples) + 1.0) * 0.5, 0, 1)
    all_images = all_images[:, :8, :, :] # clip to just 8 images
    ts, bs, channels, img_sz, img_sz = all_images.shape
    all_images = np.reshape(all_images, (ts * bs, channels, img_sz, img_sz), order='F')
    all_images = np.uint8(255 * all_images)
    imageio.imwrite('diffusion_outputs/merged_outputs_noise.png', create_image_grid(all_images, ncols=10))
    

def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    dataloader = get_data_loader(opts.data, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    sampling_loop(dataloader, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate (default 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Testing hyper-parameters
    parser.add_argument('--num_samples', type=int, default=10)

    # Data sources
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed', help='The folder of the training dataset.')
    parser.add_argument('--data_preprocess', type=str, default='deluxe', help='data preprocess scheme [basic|deluxe]')
    parser.add_argument('--use_diffaug', action='store_true', help='Use diff-augmentation during training or not')
    parser.add_argument('--ext', type=str, default='*.png', help='Choose the file type of images to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    opts.sample_dir = os.path.join('output/', opts.sample_dir,
                                   '%s_%s' % (os.path.basename(opts.data), opts.data_preprocess))
    if opts.use_diffaug:
        opts.sample_dir += '_diffaug'

    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)
    logger = SummaryWriter(opts.sample_dir)
    print(opts)
    main(opts)
