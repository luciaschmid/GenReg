from collections import OrderedDict

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule

from model.generator import Generator
from model.discriminator import Discriminator
from training import losses


class GenReg_Trainer(LightningModule):
    def __init__(self, params):
        super().__init__()

        self.save_hyperparameters()
        self.params = params

        self.G_model = Generator()
        # G_model.apply(xavier_init)
        self.G_model = nn.DataParallel(self.G_model).to(self.device)
        self.D_model = Discriminator()
        # D_model.apply(xavier_init)
        self.D_model = torch.nn.DataParallel(self.D_model).to(self.device)

        # self.G_model.train(), self.D_model.train()

        self.optimizer_D = torch.optim.Adam(self.D_model.parameters(), lr=params["lr_D"], betas=(0.9, 0.999))
        self.optimizer_G = torch.optim.Adam(self.G_model.parameters(), lr=params["lr_G"], betas=(0.9, 0.999))

        # self.D_scheduler = MultiStepLR(self.optimizer_D, [50, 80], gamma=0.2)
        # self.G_scheduler = MultiStepLR(self.optimizer_G, [50, 80], gamma=0.2)

    def forward(self, cloud_a, cloud_b):
        return self.G_model(cloud_a, cloud_b)

    def general_step(self, batch, batch_id, optimizer_idx, mode):
        cloud_a, cloud_b = batch["input"].float().to(self.device), batch["output"].float().to(self.device)
        cloud_a_g, cloud_b_g = self.G_model(cloud_a, cloud_b)

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            loss_d = 0
            for cloud_label in [(cloud_a_g, False), (cloud_a, True), (cloud_b_g, False), (cloud_b, True)]:
                cloud, label = cloud_label

                pred = self.D_model(cloud.detach())
                loss_i = losses.calc_discriminator_loss(pred, label=label)
                loss_d = loss_d + loss_i

            # discriminator loss is the average of these
            loss_d = loss_d / 4
            tqdm_dict = {"loss_d": loss_d}
            output = OrderedDict({"loss": loss_d, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        if optimizer_idx == 0:
            if batch_id % 5 == 0 or mode != "train":
                # train generator every 5 steps
                loss_g_total = self.compute_generator_loss(cloud_a, cloud_b, cloud_a_g, cloud_b_g)
                tqdm_dict = {"g_loss": loss_g_total}
                output = OrderedDict({"loss": loss_g_total, "progress_bar": tqdm_dict, "log": tqdm_dict})
                return output

            return None

    def training_step(self, batch, batch_id, optimizer_idx):
        self.G_model.train()
        self.D_model.train()
        return self.general_step(batch, batch_id, optimizer_idx, "train")

    def validation_step(self, batch, batch_id, mode='val'):
        self.G_model.eval()
        self.D_model.eval()
        return self.general_step(batch, batch_id, 0, mode)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode="test")

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True

        losses = []
        for x in outputs:
            if isinstance(x, dict):
                losses.append(x["loss"])
            else:
                for x_i in x:
                    losses.append(x_i["loss"])

        avg_loss = torch.stack(losses).mean()

        self.log(f'{mode}_loss', avg_loss, logger=False, prog_bar=True)
        if logger and self.logger is not None:
            self.logger.experiment.add_scalar(f'{mode}/{mode}_loss', avg_loss, self.current_epoch)

        return avg_loss

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss = self.general_epoch_end(outputs, 'test')
        return {
            'avg_loss': avg_loss,
        }

    def compute_generator_loss(self, cloud_a, cloud_b, cloud_a_g, cloud_b_g):

        loss_g_total = losses.calc_training_loss(cloud_a, cloud_b, cloud_a_g, cloud_b_g,
                                                         self.D_model(cloud_a), self.D_model(cloud_b_g), self.G_model)
        return loss_g_total

    def configure_optimizers(self):
        optimizer_D = torch.optim.Adam(self.D_model.parameters(), lr=self.params["lr_D"], betas=(0.9, 0.999))
        optimizer_G = torch.optim.Adam(self.G_model.parameters(), lr=self.params["lr_G"], betas=(0.9, 0.999))

        D_scheduler = MultiStepLR(optimizer_D, [50, 80], gamma=0.2)
        G_scheduler = MultiStepLR(optimizer_G, [50, 80], gamma=0.2)

        return [optimizer_G, optimizer_D], [G_scheduler, D_scheduler]
