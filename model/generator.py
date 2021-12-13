import torch.nn as nn
from pointmixer import PointMixer
from featureinteraction import FeatureInteraction
from decoder import Decoder
from pdsac import PDSAC


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

    def forward(self, x):
        # TODO
        return x
