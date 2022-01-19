import torch.nn as nn
import torch
import pytorch3d


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

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


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

        batch_size = cloud_a.size(dim=0)

        # MLP to extract local features
        cloud_a_feat = self.mlp1(cloud_a)
        cloud_b_feat = self.mlp1(cloud_b)

        # Max-Pooling to get global features
        cloud_a_max = torch.max(cloud_a_feat, 2)[0].contiguous()

        if self.two_pooling:  # get global feature for both point clouds
            cloud_b_max = torch.max(cloud_b_feat, 2)[0].contiguous()
            # concatenate both global features
            concatenated = torch.cat([cloud_a_max, cloud_b_max], dim=1)
        else:  # like in MaskNet paper
            cloud_a_max = cloud_a_max.unsqueeze(2)
            cloud_a_max = cloud_a_max.repeat(1, 1, cloud_b.size(dim=2))
            concatenated = torch.cat([cloud_b_feat, cloud_a_max], dim=1)

        # MLP from Interaction Module
        interaction_output = self.mlp2(concatenated)

        if not self.two_pooling:  # get average freedom parameters
            interaction_output = torch.mean(interaction_output, dim=2)

        # get rigid transformation matrix
        transform = pytorch3d.transforms.euler_angles_to_matrix(interaction_output[:, :3])
        bottom_row = torch.zeros(batch_size, 3).unsqueeze(2)
        transform = torch.cat([transform, bottom_row], dim=1)
        right_column = torch.cat([interaction_output[:, 3:], torch.ones(batch_size).unsqueeze(1)], dim=1)
        transform = torch.cat([transform, right_column], dim=2)

        # homogeneous coordinates for point cloud A
        cloud_a_homo = torch.cat([cloud_a, torch.ones(batch_size, cloud_a.size(dim=2)).unsqueeze(1)], dim=-1)
        # transform point cloud
        cloud_a_transformed = torch.einsum('bfg,bgn->bfn', transform, cloud_a_homo)

        return cloud_a_transformed


class GCNN(nn.Module):  # Graph convolution neural network
    def __init__(self):
        super().__init__()

        # convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(negative_slope=0.2)  # Leaky ReLU activation function
        )

    def forward(self, cloud_a, cloud_b):
        # graph construction layer
        cloud_a = get_graph_feature(cloud_a)
        cloud_b = get_graph_feature(cloud_b)
        # convolutional layer
        cloud_a = self.conv(cloud_a)
        cloud_b = self.conv(cloud_b)

        return cloud_a, cloud_b


class MixerMLP(nn.Module):
    def __init__(self, neurons, in_feat):
        super.__init__()
        self.lin1 = nn.Linear(in_features=in_feat, out_features=neurons)
        self.lin2 = nn.Linear(in_features=neurons, out_features=neurons)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.lin2(self.act(self.lin1(x)))
        return output


class MixerLayer(nn.Module):
    def __init__(self):
        super.__init__()
        self.layer_norm_token = nn.LayerNorm(6)
        self.token_mixing = MixerMLP(1024, 6)
        self.layer_norm_channel = nn.LayerNorm(1024)
        self.channel_mixing = MixerMLP(128, 1024)

    def forward(self, x):
        assert x.size(dim=-1) == 6, 'Input of Mixer Layer should be of dimension N*6'
        x1 = self.layer_norm_token(x)
        # Token Mixing on point dimension
        x1 = x1.permute(0, 2, 1)  # go to point dimension
        x1 = self.token_mixing(x1)  # perform token mixing
        # swap back to channel dimension
        x1 = x1.permute(0, 2, 1)
        x = x + x1
        # Channel Mixing on channel dimension
        x1 = self.layer_norm_channel(x)
        x = x + self.channel_mixing(x1)
        assert x.size(dim=-1) == 128, 'Output of Mixer Layer should be of dimension N*128'
        return x


class PointMixer(nn.Module):
    def __init__(self, two_pooling=True):
        super().__init__()

        self.tnet = TNet(two_pooling=two_pooling)
        self.gcnn = GCNN()
        self.mixer_layer_1 = MixerLayer()
        self.mixer_layer_2 = MixerLayer()

    def forward(self, cloud_a, cloud_b):
        cloud_a_transformed = self.tnet(cloud_a, cloud_b)
        graph_a, graph_b = self.gcnn(cloud_a_transformed, cloud_b)
        mixer_a = self.mixer_layer_2(self.mixer_layer_1(graph_a))
        mixer_b = self.mixer_layer_2(self.mixer_layer_1(graph_b))
        return mixer_a, mixer_b

