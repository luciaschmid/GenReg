from pathlib import Path
import torch
from model.generator import Generator
from model.discriminator import Discriminator



def train(model, train_dataloader, device, config):
    # TODO
    pass


def main(config):
    """
    Function for training GenReg
    :param config: configuration for training - has the following keys
        TODO add keys
    """

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # TODO create dataloaders
    train_dataset = None
    train_dataloader = None

    # TODO Instantiate model
    model = None

    # TODO Optionally Load model if resuming from checkpoint

    # Move model to specified device
    model.to(device)

    # TODO Create folder for saving checkpoints

    # Start training
    train(model, train_dataloader, device, config)
