from torch import nn
from model.pointmixer import PointMixer
from model.featureinteraction import FeatureInteraction
from model.decoder import Decoder

class GenReg(nn.Module):

    def __init__(self):
        super().__init__()

        self.pointmixer = PointMixer(two_pooling=False)
        self.feature_interaction = FeatureInteraction()
        self.decoder = Decoder(256)

    def forward(self, A, B):

        cloud_a, cloud_b = A.float(), B.float()
        pm = PointMixer(two_pooling=False)
        mixer_a, mixer_b = pm(cloud_a, cloud_b)
        
        feat1, feat2 = mixer_a, mixer_b
        feature_interaction = FeatureInteraction()
        fusion1, fusion2 = feature_interaction(feat1, feat2)
        
        x = fusion1 
        decoder = Decoder(256)
        Ag = decoder(x)
        
        x = fusion2 
        decoder = Decoder(256)
        Bg = decoder(x)

        return Ag, Bg
