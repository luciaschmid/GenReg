import torch.nn as nn
import torch

class PDSAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, T_a, T_b):
        return pdsac(T_a, T_b)

def pdsac(T_a, T_b):
    return torch.inverse(T_a) @ T_b

if __name__ == "__main__":
    T_a, T_b = torch.rand(4, 4), torch.rand(4, 4)
    T_e = pdsac(T_a, T_b)
    print("output shape is", T_e.shape)
    assert T_e.shape == (4, 4)