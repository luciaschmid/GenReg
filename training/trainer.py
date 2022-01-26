import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import datetime, time

from model.generator import Generator
from model.discriminator import Discriminator
from training import losses

class Trainer:
    def __init__(self, params, dataloaders, num_workers=12):
        self.params = params
        self.num_workers = num_workers
        self.trainloader, self.valloader, self.testloader = dataloaders['train'], dataloaders['val'], dataloaders[
            'test']

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using device: cuda')
        else:
            print('Using CPU')

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

        self.best_loss_g = 100
        self.start_t, self.epoch_n = 0, 0
        self.patience = self.params["patience"]

    def train(self):
        self.start_t = time.time()
        for e in range(self.params["n_epochs"]):
            self.epoch_n = e
            mean_loss_d, mean_loss_g = self.train_one_epoch()
            self.eval_one_epoch()

            if self.patience == 0:
                print("patience reached...")
                break

        mean_loss_d, mean_loss_g = self.test_one_epoch()
        print(f"test discriminator and generator losses are: {mean_loss_d}, {mean_loss_g}")

    def train_one_epoch(self):
        self.G_model.train()
        self.D_model.train()

        return self.general_epoch_step("train")

    # todo: implement save best checkpoint
    def do_on_best(self):
        print(f"Best epoch with total generator loss of {self.best_loss_g:2.4f}")

    def eval_one_epoch(self, epoch_n, start_t):
        self.G_model.test()
        self.D_model.test()
        mean_loss_d, mean_loss_g = self.general_epoch_step("val")

        if mean_loss_g < self.best_loss_g:
            self.best_loss_g = mean_loss_g
            self.do_on_best()
        else:
            self.patience -= 1

        return mean_loss_d, mean_loss_g

    def test_one_epoch(self):
        self.G_model.test()
        self.D_model.test()
        return self.general_epoch_step("test")

    def compute_generator_loss(self, cloud_a, cloud_b, cloud_a_g, cloud_b_g):
        loss_g_total = losses.calc_training_loss(cloud_a, cloud_b, cloud_a_g, cloud_b_g,
                                                 self.D_model(cloud_a), self.D_model(cloud_b_g), self.G_model)
        return loss_g_total

    def general_epoch_step(self, mode):
        if mode == "train":
            self.D_scheduler.step()
            self.G_scheduler.step()
            loader = self.trainloader
        elif mode == "val":
            loader = self.valloader
        else:
            loader = self.testloader

        mean_loss_g, mean_loss_d = [], []

        for batch_id, batch in enumerate(loader):
            start_t_batch = time.time()
            if mode == "train":
                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()

            cloud_a, cloud_b = batch["input"].float().to(self.device), batch["output"].float().to(self.device)
            # t_real = batch["transform"]
            cloud_a_g, cloud_b_g = self.G_model(cloud_a, cloud_b)

            loss_d = 0
            for cloud_label in [(cloud_a_g, False), (cloud_a, True), (cloud_b_g, False), (cloud_b, True)]:
                cloud, label = cloud_label

                pred = self.D_model(cloud.detach())
                loss_i = losses.calc_discriminator_loss(pred, label=label)
                if mode == "train":
                    loss_i.backward()
                    self.optimizer_D.step()
                loss_d = loss_d + loss_i.cpu().item()

            mean_loss_d.append(loss_d)

            if batch_id % 5 == 0:
                # train generator every 5 steps
                loss_g_total = self.compute_generator_loss(cloud_a, cloud_b, cloud_a_g, cloud_b_g)
                if mode == "train":
                    loss_g_total.backward()
                    self.optimizer_G.step()

                mean_loss_g.append(loss_g_total.cpu().item())
                if mode == "train":
                    msg = "{:0>8},{}:{}, [{}/{}], {}: {},{}: {},{}:{}".format(
                        str(datetime.timedelta(seconds=round(time.time() - self.start_t))),
                        "epoch",
                        self.epoch_n,
                        batch_id + 1,
                        len(self.trainloader),
                        "total_G_loss",
                        loss_g_total.cpu().item(),
                        "D_loss",
                        loss_d,
                        "iter time",
                        (time.time() - start_t_batch)
                    )
                    print(msg)

        mean_loss_d, mean_loss_g = torch.mean(torch.tensor(mean_loss_d)), torch.mean(torch.tensor(mean_loss_g))

        return mean_loss_d, mean_loss_g


