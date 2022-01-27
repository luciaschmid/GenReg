import os

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from training.trainer import GenReg_Trainer
from data.ModelNet40 import ModelNet40DataModule
# todo add data module to SevenScenes
from data.SevenScenes import SevenScenes

datasets = {"ModelNet40": ModelNet40DataModule,
            "SevenScenes": SevenScenes, }


def get_callbacks(params):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss_g",
        patience=params["patience"],
        strict=True,
        mode="min"
    )

    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss_g",
        dirpath=os.path.join(cwd, "build", "checkpoints"),
        filename=params["experiment_name"],
        save_top_k=1,
        mode="min",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    return [
        early_stop_callback,
        checkpoint_callback,
        lr_monitor,
    ]


@hydra.main(config_path="conf", config_name="config")
def train(config):
    """
        Function for training GenReg
        :param config: configuration for training - has the following keys
            TODO add keys
        """
    trainer_params = config["trainer_params"]
    dataset_params = config["dataset_params"]
    dataset_select = dataset_params["select"]

    dataloader = datasets[dataset_select](batch_size=dataset_params["batch_size"])
    dataloader.setup()

    print(len(dataloader.train_dataloader()), len(dataloader.val_dataloader()), len(dataloader.test_dataloader()))

    callbacks = get_callbacks(config["callback_params"])

    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    print(os.path.join(cwd, "build", "log"))

    stats_logger = TensorBoardLogger(
        os.path.join(cwd, "build", "log"),
        name=config["callback_params"]["experiment_name"]
    )

    model_trainer = GenReg_Trainer(trainer_params)
    trainer = pl.Trainer(check_val_every_n_epoch=1,
                         fast_dev_run=False,
                         max_epochs=trainer_params["n_epochs"],
                         gpus=1 if torch.cuda.is_available() else 0,
                         logger=stats_logger,
                         callbacks=callbacks,
                         )
    trainer.fit(model_trainer, dataloader.train_dataloader(), dataloader.val_dataloader())


if __name__ == "__main__":
    train()
