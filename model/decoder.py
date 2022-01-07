import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            ConvGELU(in_channels, 256),
            ConvGELU(256, 128),
            ConvGELU(128, 64),
            ConvGELU(64, 3),
        )

    def forward(self, x):
        return self.model(x)


class ConvGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    x = torch.rand((2, 256, 10)) # (batches, channels, number of points)
    decoder = Decoder(256)
    output = decoder(x)
    print("output shape is ", output.shape)
    assert output.shape == (2, 3, 10)