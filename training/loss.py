from turtle import forward
import torch
import torch.nn as nn
import torch.linalg as la

class GenRegLoss(nn.Module):
    
    def __init__(self, reduction='mean'):
        super().__init__()

        self.reduction = reduction
        self.sqrt_loss = nn.MSELoss(reduction=self.reduction)
        self.mae_loss = nn.L1Loss(reduction=self.reduction)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self,A,Ag,B,Bg):
        return self.absolute_loss(A, Ag, B, Bg) + self.relative_loss(A, Ag, B, Bg)
    
    def sqrt_loss_fn(self,a,b):
        loss = torch.sqrt(self.sqrt_loss(a,b))
        return loss
    
    def MAE_loss_fn(self,a,b):
        return self.mae_loss(a,b)
    
    def calc_edge_set(self,point_cloud):
        """
        Calculate edge sets, | P_i - P_(i+1)%(N-1) |2 with N total point number of point cloud P
        :param point_cloud: input point cloud
        :return: edge set
        """
        # to calculate including the difference from last point to first point, append the first point coordinates
        # to point cloud using torch.cat
        # torch diff uses P_(i+1) - P_(i), but this does not matter here as second order norm uses power of 2,
        # therefore changed signs do not matter
        diff_tensor = torch.diff(torch.cat((point_cloud, point_cloud[0].unsqueeze(0)), dim=0), dim=0)
        edge_set = la.norm(diff_tensor, ord=2, dim=1)
        return edge_set
    
    def absolute_loss(self, A, Ag, B, Bg):
        return self.sqrt_loss_fn(A,Bg) + self.sqrt_loss_fn(B,Ag)

    def relative_loss(self,A, Ag, B, Bg):
        Ea = self.calc_edge_set(A)
        Eag = self.calc_edge_set(Ag)
        Eb = self.calc_edge_set(B)
        Ebg = self.calc_edge_set(Bg)
        
        return self.MAE_loss_fn(Ea,Eag) + self.MAE_loss_fn(Eb,Ebg)