# from https://github.com/hmgoforth/PointNetLK/blob/master/ptlk/invmat.py
""" inverse matrix """

import torch


def batch_inverse(x):
    """ M(n) -> M(n); x -> x^-1 """
    batch_size, h, w = x.size()
    assert h == w
    y = torch.zeros_like(x)
    for i in range(batch_size):
        y[i, :, :] = x[i, :, :].inverse()
    return y

def batch_inverse_dx(y):
    """ backward """
    batch_size, h, w = y.size()
    assert h == w
    yl = y.repeat(1, 1, h).view(batch_size*h*h, h, 1)
    yr = y.transpose(1, 2).repeat(1, h, 1).view(batch_size*h*h, 1, h)
    dy = - yl.bmm(yr).view(batch_size, h, h, h, h)

    return dy

class InvMatrix(torch.autograd.Function):
    """ M(n) -> M(n); x -> x^-1.
    """
    @staticmethod
    def forward(ctx, x):
        y = batch_inverse(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors   # v0.4
        batch_size, h, w = y.size()
        assert h == w

        dy = batch_inverse_dx(y)  # dy(j,k,m,n) = dy(j,k)/dx(m,n)
        go = grad_output.contiguous().view(batch_size, 1, h*h)  # [1, (j*k)]
        ym = dy.view(batch_size, h*h, h*h)  # [(j*k), (m*n)]
        r = go.bmm(ym)  # [1, (m*n)]
        grad_input = r.view(batch_size, h, h)  # [m, n]

        return grad_input

