""" gives some transform methods for 3d points
Code from https://github.com/XiaoshuiHuang/fmr/blob/master/se_math/transforms.py"""
import torch
import torch.utils.data
import data.so3_se3 as so3_se3


class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)


class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0]  # [N, D] -> [D]
        s = torch.max(c)  # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        # return self.method1(tensor)
        return self.method2(tensor)


class Resampler:
    """ [N, D] -> [M, D] """

    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        num_points, dim_p = tensor.size()
        out = torch.zeros(self.num, dim_p).to(tensor)

        selected = 0
        while selected < self.num:
            remainder = self.num - selected
            idx = torch.randperm(num_points)
            sel = min(remainder, num_points)
            val = tensor[idx[:sel]]
            out[selected:(selected + sel)] = val
            selected += sel
        return out


class RandomTransformSE3:
    """ rigid motion """

    def __init__(self):
        self.mag = 0.8  # magnitude
        self.randomly = True

        self.gt = None
        self.transformation_matrix = None

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        '''a = torch.rand(3)
        a = a * math.pi
        b = torch.zeros(1, 6)
        b[:, 0:3] = a
        x = x+b
        '''
        return x  # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = so3_se3.exp(x).to(p0)  # [1, 4, 4]
        gt = so3_se3.exp(-x).to(p0)  # [1, 4, 4]

        p1 = so3_se3.transform(g, p0)
        self.gt = gt.squeeze(0)  # gt: p1 -> p0
        self.transformation_matrix = g.squeeze(0)  # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)

# EOF
