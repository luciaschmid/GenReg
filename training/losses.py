import torch
import torch.nn as nn
import torch.linalg as la
from emd import earth_mover_distance


def calc_training_loss(pc1, pc2, apc1, apc2, real_o, real_t, fake_o, fake_t, transf_est, transf_gt):
    training_loss = calc_absolute_loss(pc1, pc2, apc1, apc2) + calc_relative_loss(pc1, pc2, apc1, apc2) +\
                    calc_cycle_consistency_loss(pc1, pc2, None, None) +\
                    calc_adversarial_loss(real_o, real_t, fake_o, fake_t) * 0.01 +\
                    calc_transformation_loss(transf_est, transf_gt)
    return training_loss


def calc_absolute_loss(pointcloud_a, pointcloud_b, aligned_pointcloud_a, aligned_pointcloud_b):
    if pointcloud_a.size(dim=-1) == 3:
        use_transpose = False
    elif pointcloud_a.size(dim=-2) == 3:
        use_transpose = True
    else:
        raise ValueError("Point Cloud should have dimension 3 (x,y,z of points) at last or second to last position.")

    absolute_loss = earth_mover_distance(pointcloud_a, aligned_pointcloud_b, transpose=use_transpose) +\
        earth_mover_distance(pointcloud_b, aligned_pointcloud_a, transpose=use_transpose)
    return absolute_loss


def calc_edge_set(point_cloud):
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


def calc_relative_loss(pointcloud_a, pointcloud_b, aligned_pointcloud_a, aligned_pointcloud_b):
    # calculate edge sets, | P_i - P_(i+1)%(N-1) |2 with N total point number of point cloud P
    edge_a = calc_edge_set(pointcloud_a)
    edge_aligned_a = calc_edge_set(aligned_pointcloud_a)
    edge_b = calc_edge_set(pointcloud_b)
    edge_aligned_b = calc_edge_set(aligned_pointcloud_b)

    # calculate relative loss
    relative_loss = nn.L1Loss(edge_a, edge_aligned_a) + nn.L1Loss(edge_b, edge_aligned_b)
    return relative_loss


def calc_cycle_consistency_loss(pointcloud_a, pointcloud_b, F, G):
    # ToDO add cycle consistency loss
    return 10


def calc_adversarial_loss(real_output, real_truth, fake_output, fake_truth):
    criterion = nn.BCELoss()
    real_loss = criterion(real_output, real_truth)
    fake_loss = criterion(fake_output, fake_truth)
    return real_loss + fake_loss


def calc_transformation_loss(transformation_estimated, transformation_ground_truth):
    transformation_product = torch.matmul(transformation_estimated, la.inv(transformation_ground_truth))
    transformation_loss = torch.dist(transformation_product, torch.eye(4), p=2)
    return transformation_loss


if __name__ == "__main__":
    p1 = torch.rand(1024, 3).to(device='cuda')
    p2 = torch.rand(1024, 3).to(device='cuda')
    emd_p1_p2 = earth_mover_distance(p1, p2, transpose=False)
    print(f'EMD is {emd_p1_p2}')
    edge_set_p1 = calc_edge_set(p1)
    print(f'Edge set of p1 is {edge_set_p1} and shape {edge_set_p1.shape}')
