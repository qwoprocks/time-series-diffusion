# CMU 16-726 Learning-Based Image Synthesis / Spring 2024, Assignment 3
#
# Usage:
# ======
#    To train with the default hyperparamters:
#       python train_ddpm.py

import argparse
import os
import warnings

import imageio

from diff_augment import DiffAugment
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
from diffusion_model import Unet, p_losses, sample
import matplotlib.pyplot as plt


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

def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    U = create_model(opts)

    # Create optimizers for the generators and discriminators
    u_optimizer = optim.Adam(U.parameters(), opts.lr, [opts.beta1, opts.beta2])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(opts.num_epochs):
        print(f"--- Epoch [{epoch}/{opts.num_epochs}] ---")

        for step, batch in enumerate(train_dataloader):
            real_images = batch
            real_images = utils.to_var(real_images)
            real_images = real_images.to(device)

            #######################################
            ###         TRAIN THE UNET         ####
            #######################################

            # FILL THIS IN
            # 1. Sample t uniformally for every example in the batch
            t = torch.randint(low=0, high=opts.denoising_steps, size=(real_images.shape[0],), device=device).long()

            # 2. Get loss between loss and predicted loss
            loss = p_losses(U, real_images, t, loss_type='l2')

            if step % 100 == 0:
                print("Loss:", loss.item())

            u_optimizer.zero_grad()
            loss.backward()
            u_optimizer.step()
        
        if epoch % 9 == 0:
            print("Saving...")
            torch.save(U.state_dict(), "diffusion.pth")

def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    dataloader = get_data_loader(opts.data, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    training_loop(dataloader, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate (default 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Testing hyper-parameters
    parser.add_argument('--denoising_steps', type=int, default=500)

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
