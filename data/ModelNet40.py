from pathlib import Path
import torch


class ModelNet40(torch.utils.data.Dataset):
    dataset_path = Path("data/ModelNet40")

    def __init__(self):
        super().__init__()
        # TODO

    def __getitem__(self, item):
        # TODO
        pass

    def __len__(self):
        # TODO
        return None

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO
        pass
