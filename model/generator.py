from torch import nn
import torch
from model.pointmixer import PointMixer
from model.featureinteraction import FeatureInteraction
from model.decoder import Decoder
from model.pdsac import PDSAC


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO add parameters to calls

        self.encoder = nn.Sequential(
            PointMixer(),
            FeatureInteraction()
        )
        self.decoder = Decoder()
        self.pdsac = PDSAC()

    def forward(self, cloud_a, cloud_b):

        fusion1, fusion2 = self.encoder(cloud_a, cloud_b)
        Ag = self.decoder(fusion1)
        Bg = self.decoder(fusion2)
        t_a = self.pdsac(cloud_a, Ag)
        t_b = self.pdsac(cloud_b, Bg)
        t_e = torch.inverse(t_a) @ t_b
        return Ag, Bg, t_e
