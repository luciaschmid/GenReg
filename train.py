import hydra
from torch.utils.data.dataloader import DataLoader

from training.trainer import Trainer
from data.ModelNet40 import ModelNet40
from data.SevenScenes import SevenScenes
datasets = {"ModelNet40": ModelNet40,
            "SevenScenes": SevenScenes,}

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

    dataset = datasets[dataset_select]
    train_dataset = dataset('train')
    test_dataset = dataset('test')
    cross_dataset = dataset('cross-category')

    dataloaders = {"train": DataLoader(train_dataset, batch_size=dataset_params["batch_size"], shuffle=False),
                   "val": DataLoader(test_dataset, batch_size=dataset_params["batch_size"], shuffle=False),
                   "test": DataLoader(cross_dataset, batch_size=dataset_params["batch_size"], shuffle=False)}

    print(len(dataloaders["train"]), len(dataloaders["val"]), len(dataloaders["test"]))

    trainer = Trainer(trainer_params, dataloaders)
    trainer.train()


if __name__ == "__main__":
    train()