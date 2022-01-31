from pathlib import Path

import numpy as np
import torch

from model.genreg import GenReg
from data.ModelNet40 import ModelNet40
from training.loss import GenRegLoss


def train(model, train_dataloader, val_dataloader, device, config):

    loss_criterion = GenRegLoss()

    # TODO: Declare optimizer with learning rate given in config
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # TODO: Set model to train
    model.train()

    best_loss_val = np.inf

    # Keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # TODO: Move batch to device, set optimizer gradients to zero, perform forward pass
            # ShapeNet.move_batch_to_device(batch, device)

            optimizer.zero_grad()

            A = batch['input'].float()
            B = batch['output'].float()

            Ag, Bg = model(A,B)
            

            # TODO: Compute loss, Compute gradients, Update network parameters
            loss = loss_criterion(A, Ag, B, Bg)
            loss.backward()
            optimizer.step()

            # Logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # TODO: Set model to eval

                model.eval()

                # Evaluation on entire validation set
                loss_val = 0.
                for batch_val in val_dataloader:
                    # ShapeNet.move_batch_to_device(batch_val, device)

                    with torch.no_grad():
                        A = batch_val['input']
                        B = batch_val['output']

                        Ag, Bg = model(A,B)

                        

                    loss_val += loss_criterion(A, Ag, B, Bg).item()

                loss_val /= len(val_dataloader)
                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'./runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

                # TODO: Set model back to train
                model.train()

        scheduler.step()


def main(config):
    

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ModelNet40('train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=train_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    val_dataset = ModelNet40('cross-category')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=val_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    # Instantiate model
    model = GenReg()

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    print("its set")

    # Create folder for saving checkpoints
    # Path(f'./runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
