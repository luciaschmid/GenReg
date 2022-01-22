import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
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
        self.model = nn.Sequential(
            LinearLeakyReLU(in_features=in_features, out_features=512),
            LinearLeakyReLU(in_features=512, out_features=256),
            LinearLeakyReLU(in_features=256, out_features=1),
        )

    def forward(self, x):
        return self.model(x)


class LinearLeakyReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    x = torch.rand((2, 256*3))
    discriminator = Discriminator(in_features=256*3)
    output = discriminator(x)
    print("output shape is ", output.shape)
    assert output.shape == (2, 1)
