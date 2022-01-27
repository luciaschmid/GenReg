from pathlib import Path
import torch
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

NUM_WORKERS = int(os.cpu_count() / 2)

class ModelNet40(torch.utils.data.Dataset):
    dataset_path = os.path.join(os.getcwd(), "data/ModelNet40")

    def __init__(self,split):
        super().__init__()
        # TODO
        
        assert split in ['train', 'cross-category', 'test']
        
        self.items = list()
        
        classes = list(os.walk(self.dataset_path))[0][1]
        classes.sort()
        
        same_categories = classes[:len(classes)//2]
        cross_categories = classes[len(classes)//2:]
        
        self.s = same_categories
        self.c = cross_categories
        
        if split == "cross-category":
            for cls in cross_categories:
                files = list(os.walk(self.dataset_path+"/{}/test/".format(cls)))[0][2]
                for file in files:
                    self.items.append(cls+"/test/"+file)
        else:
            for cls in same_categories:
                files = list(os.walk(self.dataset_path+"/{}/{}/".format(cls,split)))[0][2]
                for file in files:
                    self.items.append(cls+"/"+split+"/"+file)
        
        # self.items = self.items[:100]

    def __getitem__(self, index):
        # TODO
        classname, split, off_file = self.items[index].split("/")
        input_v, output_v = self.get_object_files(classname, split, off_file)
        
        return {
            "input": input_v.T,
            "output": output_v.T,
            "class": classname
        }

    def __len__(self):
        # TODO
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO
        batch['input'] = batch['input'].to(device)
        batch['output'] = batch['output'].to(device)
    
    @staticmethod
    def normalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    @staticmethod
    def rigid_transform(mat,x_deg,y_deg,z_deg,tx,ty,tz):
        
        Rx = np.array([
            [1., 0., 0.],
            [0., np.round(np.cos(x_deg),3), np.round(-1*np.sin(x_deg),3)],
            [0., np.round(np.sin(x_deg),3), np.round(np.cos(x_deg),3)]
        ])
        
        Ry = np.array([
            [np.round(np.cos(y_deg),3), 0., np.round(np.sin(y_deg),3)],
            [0., 1., 0.],
            [np.round(-1*np.sin(y_deg),3), 0., np.round(np.cos(y_deg),3)]
        ])
        
        Rz = np.array([
            [np.round(np.cos(z_deg),3), np.round(-1*np.sin(z_deg),3), 0.],
            [np.round(np.sin(z_deg),3), np.round(np.cos(z_deg),3), 0.],
            [0., 0., 1.]
        ])
        
        translate = np.array([
            [tx],
            [ty],
            [tz]
        ])
        
        op = Rz@Ry@Rx@mat.T
        
        a = op + translate
        
        
        return a.T
        
    
    def get_object_files(self,classname,split,off_file):
        #reading data file
        f = open(self.dataset_path+"/{}/{}/{}".format(classname,split,off_file),"r")
        obj = [i.strip() for i in f.readlines()]
        f.close()
        
        
        if len(obj[0].split()) == 1:
            obj = obj[1:]
        else:
            obj[0] = obj[0].replace("OFF","")
        
        #loading vertices
        vertices_t, _, _ = map(int,obj[0].split())
        vertices = np.array([list(map(float,i.split())) for i in obj[1:vertices_t+1]])
        
        #random sampling 1024 vertices
        l = vertices.shape[0]
        indices = np.random.choice([i for i in range(l)],1024,True)
        input_v = ModelNet40.normalizeData(vertices[indices,:])
        
        #random initialization of 6DoF
        x_deg = np.random.uniform(np.deg2rad(0),np.deg2rad(45)) #x-axis Rotation
        y_deg = np.random.uniform(np.deg2rad(0),np.deg2rad(45)) #y-axis Rotation
        z_deg = np.random.uniform(np.deg2rad(0),np.deg2rad(45)) #z-axis Rotation
        
        tx = np.round(np.random.uniform(0,0.8),3) #x-axis Translation
        ty = np.round(np.random.uniform(0,0.8),3) #y-axis Translation
        tz = np.round(np.random.uniform(0,0.8),3) #z-axis Translation
        
        output_v = ModelNet40.rigid_transform(input_v.copy(),x_deg,y_deg,z_deg,tx,ty,tz)
        
        return input_v, output_v
    
    
class ModelNet40DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers: int = NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ModelNet40('train')
        self.test_dataset = ModelNet40('test')
        self.cross_dataset = ModelNet40('cross-category')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.cross_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False)