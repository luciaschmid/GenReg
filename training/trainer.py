import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from model.generator import Generator
from model.discriminator import Discriminator
from utils.invmat import InvMatrix

class Trainer:
    def __init__(self, params, dataloaders, num_workers=12):

        self.num_workers = num_workers
        self.trainloader, self.valloader, self.testloader = dataloaders['train'], dataloaders['val'], dataloaders[
            'test']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.G_model = Generator()
        # G_model.apply(xavier_init)
        self.G_model = nn.DataParallel(self.G_model).to(self.device)
        self.D_model = Discriminator()
        # D_model.apply(xavier_init)
        self.D_model = torch.nn.DataParallel(self.D_model).to(self.device)

        self.G_model.train(), self.D_model.train()

        self.optimizer_D = torch.optim.Adam(self.D_model.parameters(), lr=params["lr_D"], betas=(0.9, 0.999))
        self.optimizer_G = torch.optim.Adam(self.G_model.parameters(), lr=params["lr_G"], betas=(0.9, 0.999))

        self.D_scheduler = MultiStepLR(self.optimizer_D, [50, 80], gamma=0.2)
        self.G_scheduler = MultiStepLR(self.optimizer_G, [50, 80], gamma=0.2)
        # todo: create the loss class
        self.loss = Loss()

    def train_one_epoch(self):
        self.D_scheduler.step()
        self.G_scheduler.step()

        for batch_id, batch in enumerate(self.trainloader):
            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()

            cloud_a, cloud_b = batch["input"].float().to(self.device), batch["output"].float().to(self.device)
            t_real = batch["transform"]
            cloud_a_g, cloud_b_g, t_e = self.G_model(cloud_a, cloud_b)

            loss_d = 0
            for cloud_label in [(cloud_a_g, False), (cloud_a, True), (cloud_b_g, False), (cloud_b, True)]:
                cloud, label = cloud_label
                pred = self.D_model(cloud.detach())
                loss_i = self.loss.get_discriminator_loss_single(pred, label=label)
                loss_i.backward()
                self.optimizer_D.step()
                loss_d = loss_d + loss_i

            if batch_id % 5 == 0:
                #train generator every 5 steps
                loss_g_total = self.compute_generator_loss(cloud_a, cloud_b, cloud_a_g, cloud_b_g, t_e, t_real)
                loss_g_total.backward()
                self.optimizer_G.step()

        return loss_g, loss_d

    #todo: implement
    def general_epoch_step(self, mode):
        ...

    def eval_one_epoch(self):
        for batch_id, batch in enumerate(self.valloader):
            self.general_epoch_step("val")

    def test_one_epoch(self):
        for batch_id, batch in enumerate(self.testloader):
            self.general_epoch_step("test")

    def compute_generator_loss(self, cloud_a, cloud_b, cloud_a_g, cloud_b_g, t_e, t_real):
        # todo: implement losses
        # loss alignment
        l_abs = self.loss.emd(cloud_a, cloud_b_g) + self.loss.emd(cloud_b, cloud_b_g)
        # loss relative struction
        l_relative = self.loss.mae(cloud_a, cloud_a_g) + self.loss.mae(cloud_a, cloud_a_g)
        # reverse deformation
        l_cyc = ...
        # adversarial loss
        l_adv = ...
        # transformation loss
        l_t = torch.nn.MSELoss()(t_e.pmm(InvMatrix(t_real)), torch.eye(4))

        loss_g_total = l_abs + l_relative + l_cyc + 0.01 * l_adv + l_t
        return loss_g_total


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
