from pathlib import Path
import torch
import torchvision
import os
import glob
import data.transforms as transforms
import data.mesh as mesh

# Code adapted from https://github.com/XiaoshuiHuang/fmr/blob/master/data/dataset.py, as GenReg uses very similar
# 7 Scenes dataset setup as FMR


class SevenScenes(torch.utils.data.Dataset):
    dataset_path = Path("data/7scene")

    def __init__(self, split):
        super().__init__()
        dataset_path = Path("data/7scene")

        if not os.path.exists(SevenScenes.dataset_path):
            # using same download link as FMR github, because mentioned in GenReg paper
            exit("Please download 7Scenes files using\
                 https://drive.google.com/u/0/uc?id=1XdQ3muo5anFA28ZFch06z_iTEjXxLEQi&export=download\
                 into data/7scene")

        assert split in ['train', 'test']

        self.scenes = Path(f"data/7scene_{split}.txt").read_text().splitlines()

        self.samples = self.get_samples(self.scenes)

        self.loader = mesh.plyread

        num_points = 2048 if split == 'train' else 10000
        self.transform = torchvision.transforms.Compose([
                transforms.Mesh2Points(),
                transforms.OnUnitCube(),
                transforms.Resampler(num_points)])

        self.rigid_transform = transforms.RandomTransformSE3()

    def __getitem__(self, item):
        path, smpl_idx = self.samples[item]
        pointcloud_a = self.loader(path)
        pointcloud_a = self.transform(pointcloud_a)

        pointcloud_b = self.rigid_transform(pointcloud_a)
        transformation_matrix = self.rigid_transform.transformation_matrix

        return {'pointcloud_a': pointcloud_a, 'pointcloud_b': pointcloud_b,
                'transformation_matrix': transformation_matrix}

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['pointcloud_a'] = batch['pointcloud_a'].to(device)
        batch['pointcloud_b'] = batch['pointcloud_b'].to(device)

    @staticmethod
    def get_samples(scenes):
        scenes = {scenes[i]: i for i in range(len(scenes))}
        samples = []
        for scene in sorted(os.listdir(SevenScenes.dataset_path)):
            # check if actual scene directory
            cur_scene = os.path.join(SevenScenes.dataset_path, scene)
            if not os.path.isdir(cur_scene):
                continue
            # check if detected possible current scene is in target scenes, and gets index
            smpl_idx = scenes.get(scene)
            if smpl_idx is None:
                continue
            # get samples from scene folder
            smpl_files = os.path.join(cur_scene, '*.ply')
            names = glob.glob(smpl_files)
            for path in sorted(names):
                item = (path, smpl_idx)
                samples.append(item)
        return samples

