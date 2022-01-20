import torch.nn as nn
import torch

class FeatureInteraction(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, feat1, feat2):
        g1, g2 = self.max_pool(feat1), self.max_pool(feat2)
        diff = g2 - g1
        fusion1, fusion2 = torch.cat([feat1, diff], dim=1), torch.cat([feat2, diff], dim=1)
        return fusion1, fusion2

if __name__ == "__main__":
    feat1, feat2 = torch.rand((2, 128, 1024)), torch.rand((2, 128, 1024))
    feature_interaction = FeatureInteraction()
    fusion1, fusion2 = feature_interaction(feat1, feat2)
    print("fusion1 and fusion2 output shapes are", fusion1.shape, fusion2.shape)
    assert fusion1.shape == (2, 256, 1024)
    assert fusion2.shape == (2, 256, 1024)