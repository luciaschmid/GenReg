""" 3-d mesh reader
Code from https://github.com/XiaoshuiHuang/fmr/blob/master/se_math/mesh.py"""
import os
import copy
import numpy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot

# used to read ply files
from plyfile import PlyData


class Mesh:
    def __init__(self):
        self._vertices = []  # array-like (N, D)
        self._faces = []  # array-like (M, K)
        self._edges = []  # array-like (L, 2)

    def clone(self):
        other = copy.deepcopy(self)
        return other

    def clear(self):
        for key in self.__dict__:
            self.__dict__[key] = []

    def add_attr(self, name):
        self.__dict__[name] = []

    @property
    def vertex_array(self):
        return numpy.array(self._vertices)

    @property
    def vertex_list(self):
        return list(map(tuple, self._vertices))

    @staticmethod
    def faces2polygons(faces, vertices):
        p = list(map(lambda face: \
                         list(map(lambda vidx: vertices[vidx], face)), faces))
        return p

    @property
    def polygon_list(self):
        p = Mesh.faces2polygons(self._faces, self._vertices)
        return p

    def plot(self, fig=None, ax=None, *args, **kwargs):
        p = self.polygon_list
        v = self.vertex_array
        if fig is None:
            fig = matplotlib.pyplot.gcf()
        if ax is None:
            ax = Axes3D(fig)
        if p:
            ax.add_collection3d(Poly3DCollection(p))
        if v.shape:
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], *args, **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig, ax

    def on_unit_sphere(self, zero_mean=False):
        # radius == 1
        v = self.vertex_array  # (N, D)
        if zero_mean:
            a = numpy.mean(v[:, 0:3], axis=0, keepdims=True)  # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        n = numpy.linalg.norm(v[:, 0:3], axis=1)  # (N,)
        m = numpy.max(n)  # scalar
        v[:, 0:3] = v[:, 0:3] / m
        self._vertices = v
        return self

    def on_unit_cube(self, zero_mean=False):
        # volume == 1
        v = self.vertex_array  # (N, D)
        if zero_mean:
            a = numpy.mean(v[:, 0:3], axis=0, keepdims=True)  # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        m = numpy.max(numpy.abs(v))  # scalar
        v[:, 0:3] = v[:, 0:3] / (m * 2)
        self._vertices = v
        return self

    def rot_x(self):
        # camera local (up: +Y, front: -Z) -> model local (up: +Z, front: +Y).
        v = self.vertex_array
        t = numpy.copy(v[:, 1])
        v[:, 1] = -numpy.copy(v[:, 2])
        v[:, 2] = t
        self._vertices = list(map(tuple, v))
        return self

    def rot_zc(self):
        # R = [0, -1;
        #      1,  0]
        v = self.vertex_array
        x = numpy.copy(v[:, 0])
        y = numpy.copy(v[:, 1])
        v[:, 0] = -y
        v[:, 1] = x
        self._vertices = list(map(tuple, v))
        return self


def plyread(filepath, points_only=True):
    # read binary ply file and return [x, y, z] array
    data = PlyData.read(filepath)
    vertex = data['vertex']

    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
    num_verts = len(x)

    mesh = Mesh()

    for v in range(num_verts):
        vp = tuple(float(s) for s in [x[v], y[v], z[v]])
        mesh._vertices.append(vp)

    return mesh


# EOF
