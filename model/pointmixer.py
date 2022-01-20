from torch import nn
from torch.nn import functional as F
import torch

from model.featureinteraction import FeatureInteraction

def get_graph_feature(x, k=20):
    """
    Code from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py, only slightly adapted to build graph
    :param x: point cloud
    :param k: number of neighbors
    :return: constructed graph
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx.to(device) + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a

def get_trans_matrix(rot, t):
    trans = torch.zeros((rot.shape[0], 4, 4))
    trans[:, :3, :3] = rot
    trans[:, :3, 3] = t
    trans[:, 3, 3] = 1
    return trans

class TNet(nn.Module):
    def __init__(self, two_pooling=True):
        super().__init__()
        self.two_pooling = two_pooling

        # First Three-Layer MLP
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.feat_interaction = FeatureInteraction()

        # Second Three-Layer MLP
        self.mlp2 = nn.Sequential(
            nn.Conv1d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 6, 1),
        )

    def forward(self, cloud_a, cloud_b):
        # Check that point clouds are in the format B x C x N (B: batch size C: channel number N: number of points)
        assert cloud_a.size(dim=1) == 3, 'second dimension of point cloud should be # of channels, therefore 3'
        assert cloud_b.size(dim=1) == 3, 'second dimension of point cloud should be # of channels, therefore 3'

        # First: MLP to extract local features
        cloud_a_feat = self.mlp1(cloud_a) # [b, 1024, 1024]
        cloud_b_feat = self.mlp1(cloud_b)

        # Second: max pooling is applied to get the global features
        # Third: the interaction module concatenates the both global features, and uses the concatenated
        # feature into a three-layer MLP to estimate the six transformation parameters
        fusion1, fusion2 = self.feat_interaction(cloud_a_feat, cloud_b_feat) # [b, 2048, 1024]
        params_a, params_b = symfn_max(self.mlp2(fusion1)), symfn_max(self.mlp2(fusion2)) #[b, 6, 1]

        # get rigid transformation matrix
        rot_a = euler_angles_to_matrix(params_a[:, :3, 0], "ZYX")
        rot_b = euler_angles_to_matrix(params_b[:, :3, 0], "ZYX")
        t_a, t_b = params_a[:, 3:, 0], params_b[:, 3:, 0]
        trans_a, trans_b = get_trans_matrix(rot_a, t_a), get_trans_matrix(rot_b, t_b)

        # homogeneous coordinates for point cloud A
        cloud_a_homo = nn.functional.pad(cloud_a, (0,0,0,1), "constant", 1.0)
        cloud_b_homo = nn.functional.pad(cloud_b, (0,0,0,1), "constant", 1.0)

        cloud_a_transformed = torch.einsum('bfg,bgn->bfn', trans_a, cloud_a_homo)
        cloud_b_transformed = torch.einsum('bfg,bgn->bfn', trans_b, cloud_b_homo)

        return cloud_a_transformed[:, :3, :], cloud_b_transformed[:, :3, :]


class GCNN(nn.Module):  # Graph convolution neural network
    def __init__(self):
        super().__init__()
        emb_dims = 128
        self.k = 20
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)

        return x


class MixerMLP(nn.Module):
    def __init__(self, neurons, in_feat):
        super().__init__()
        self.lin1 = nn.Linear(in_features=in_feat, out_features=neurons)
        self.lin2 = nn.Linear(in_features=neurons, out_features=neurons)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.lin2(self.act(self.lin1(x)))
        return output


class MixerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_token = nn.LayerNorm(128)
        self.token_mixing = MixerMLP(1024, 1024)
        self.layer_norm_channel = nn.LayerNorm(1024)
        self.channel_mixing = MixerMLP(128, 128)

    def forward(self, x):
        assert x.size(dim=1) == 128, 'Input of Mixer Layer should be of dimension N*6'
        # Token Mixing on point dimension
        x1 = x.permute(0, 2, 1)
        x1 = self.layer_norm_token(x1)
        x1 = x1.permute(0, 2, 1)
        x1 = self.token_mixing(x1)  # perform token mixing
        x = x + x1

        # Channel Mixing on channel dimension
        x1 = self.layer_norm_channel(x)
        x1 = x1.permute(0, 2, 1)
        x = x + self.channel_mixing(x1).permute(0, 2, 1)
        assert x.size(dim=1) == 128, 'Output of Mixer Layer should be of dimension N*128'
        return x


class PointMixer(nn.Module):
    def __init__(self, two_pooling=True):
        super().__init__()

        self.tnet = TNet(two_pooling=two_pooling)
        self.gcnn = GCNN()
        self.mixer_layer_1 = MixerLayer()
        self.mixer_layer_2 = MixerLayer()

    def forward(self, cloud_a, cloud_b):
        cloud_a_transformed, _ = self.tnet(cloud_a, cloud_b)
        graph_a = self.gcnn(cloud_a_transformed)
        graph_b = self.gcnn(cloud_b)

        mixer_a = self.mixer_layer_2(self.mixer_layer_1(graph_a))
        mixer_b = self.mixer_layer_2(self.mixer_layer_1(graph_b))
        return mixer_a, mixer_b

if __name__ == "__main__":
    cloud_a, cloud_b = torch.rand(2, 3, 1024), torch.rand(2, 3, 1024)
    pm = PointMixer(two_pooling=False)
    mixer_a, mixer_b = pm(cloud_a, cloud_b)
    print("output shape is", mixer_a.shape, mixer_b.shape)
