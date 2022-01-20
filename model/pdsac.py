import torch.nn as nn
import torch

# todo: needs to be edited
class PDSAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, Ag):
        ...
        return # transformation matrix

if __name__ == "__main__":
    A, Ag = torch.rand(2, 3, 1024), torch.rand(2, 3, 1024)
    pdsac = PDSAC()
    T_a = pdsac(A, Ag)
    print("output shape is", T_a.shape)
    assert T_a.shape == (4, 4)