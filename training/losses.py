import torch
import torch.nn as nn
import torch.linalg as la
from training.emd import earth_mover_distance

bce_loss = nn.BCEWithLogitsLoss()
def calc_discriminator_loss(pred, label):
    labels = torch.full((pred.size(0), 1), label, dtype=torch.float).to(pred.device)
    return bce_loss(pred, labels)

def calc_training_loss(pc1, pc2, apc1, apc2, real_o_a, fake_o_b, F, transf_est=None, transf_gt=None):
    training_loss = calc_absolute_loss(pc1, pc2, apc1, apc2) + calc_relative_loss(pc1, pc2, apc1, apc2) +\
                    calc_cycle_consistency_loss(pc1, apc1, pc2, apc2, F) +\
                    calc_adversarial_loss(real_o_a, fake_o_b) * 0.01
                    # + calc_transformation_loss(transf_est, transf_gt)
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
    return absolute_loss.mean()


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
    relative_loss = nn.L1Loss()(edge_a, edge_aligned_a) + nn.L1Loss()(edge_b, edge_aligned_b)
    return relative_loss


def calc_cycle_consistency_loss(cloud_a, cloud_a_fake, cloud_b, cloud_b_fake, F):
    # ToDO add cycle consistency loss
    re_cloud_a, re_cloud_b = F(cloud_a_fake, cloud_b_fake)
    l = earth_mover_distance(re_cloud_a, cloud_a) + earth_mover_distance(re_cloud_b, cloud_b)
    return l.mean()


def calc_adversarial_loss(real_output, fake_output):
    real_loss = calc_discriminator_loss(real_output, True)
    fake_loss = calc_discriminator_loss(fake_output, False)
    return real_loss + fake_loss


def calc_transformation_loss(transformation_estimated, transformation_ground_truth):
    transformation_product = torch.matmul(transformation_estimated, la.inv(transformation_ground_truth))
    transformation_loss = torch.dist(transformation_product, torch.eye(4), p=2)
    return transformation_loss


if __name__ == "__main__":
    p1 = torch.rand(2, 1024, 3).to(device='cuda')
    p2 = torch.rand(2, 1024, 3).to(device='cuda')
    emd_p1_p2 = earth_mover_distance(p1, p2, transpose=False)
    print(f'EMD is {emd_p1_p2}')
    edge_set_p1 = calc_edge_set(p1)
    print(f'Edge set of p1 is {edge_set_p1} and shape {edge_set_p1.shape}')

    output = torch.rand(2, 1)
    label = False
    d_loss = calc_discriminator_loss(output, label)
    print("d loss is ", d_loss)