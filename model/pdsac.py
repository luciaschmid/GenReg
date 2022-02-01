import torch.nn as nn
import torch
import math


class PDSAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, A_g, B, B_g):
        T_A = pdsac(A, A_g)
        T_B = pdsac(B, B_g)
        return torch.matmul(torch.inverse(T_A), T_B)


def pdsac(point_cloud, aligned_point_cloud, m=20):
    # center each cloud on the origin
    point_cloud_cog = torch.sum(point_cloud, dim=-1)
    c_point_cloud = point_cloud - point_cloud_cog.unsqueeze(dim=-1)
    aligned_point_cloud_cog = torch.sum(aligned_point_cloud, dim=-1)
    c_aligned_point_cloud = aligned_point_cloud - aligned_point_cloud_cog.unsqueeze(dim=-1)

    k = 4  # number of correspondences / point pairs per set

    # randomly sample m minimal sets as hypothesis inlier sets pool
    idx = torch.randperm(point_cloud.size(-1))[:m*k]
    M_pc = c_point_cloud[:, :, idx].view(-1, m, 3, k)  # batch size x m minimal sets x 3 coordinates per point x k points
    M_apc = c_aligned_point_cloud[:, :, idx].view(-1, m, 3, k)

    # Compute transformation matrices
    N = torch.einsum('bmck,bmdk -> bmcd', M_pc, M_apc)
    U, S, Vh = torch.linalg.svd(N)  # compute singular value decomposition to be able to compute rotation matrix
    rotation = torch.matmul(Vh.permute(0, 1, 3, 2), U.permute(0, 1, 3, 2))
    rotation = torch.nn.functional.pad(rotation, (0, 0, 0, 1), value=0.0)
    translation = aligned_point_cloud_cog - point_cloud_cog
    translation = torch.nn.functional.pad(translation.unsqueeze(1).repeat_interleave(repeats=m, dim=1), (0, 1),
                                          value=1.0).unsqueeze(3)
    transform = torch.cat((rotation, translation), dim=-1)

    # select transformation with the minimum projection error
    point_cloud_homo = torch.nn.functional.pad(point_cloud, (0, 0, 0, 1), value=1.0)  # homogeneous coordinates
    aligned_point_cloud_homo = torch.nn.functional.pad(aligned_point_cloud, (0, 0, 0, 1), value=1.0)
    errors = torch.linalg.norm(torch.einsum('bmcd,bdk->bmck', transform, point_cloud_homo) -
                               aligned_point_cloud_homo.unsqueeze(1),
                               dim=(-2, -1))
    index_min = torch.argmin(errors, dim=-1)
    index_min = index_min.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat_interleave(4, 2).repeat_interleave(4, 3)
    best_transform = torch.gather(transform, dim=1, index=index_min)
    return best_transform.squeeze(1)


if __name__ == "__main__":
    pc = torch.rand(4, 3, 80)
    pc_homo = torch.nn.functional.pad(pc, (0, 0, 0, 1), value=1.0)
    T = torch.tensor([[math.sqrt(3)/2, -0.5, 0, 2], [0.5, math.sqrt(3)/2, 0, 1], [0, 0, 1, 2], [0, 0, 0, 1]])
    pc_a_homo = torch.matmul(T, pc_homo)
    pc_a = pc_a_homo[:, :3, :]
    print("Aligned Point Cloud Shape is ", pc_a.shape)
    T_est = pdsac(pc, pc_a)
    print(f"Real Transformation matrix is {T}")
    print(f"Estimated Transformation matrix is {T_est}")
