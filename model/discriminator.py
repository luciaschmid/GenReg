import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        '''
        From Appendix Section 1.1
        Input N*3
        FC(N*3, 512)
        LeakyReLU(0.2)
        FC(512, 256)
        LeakyReLU(0.2)
        FC(256, 1)
        LeakyReLU(0.2)
        Output Fake(0)/Real(1)
        '''

    def forward(self, x):
        # TODO
        return x
