import torch

from _utils import gen_test_data


def adj_normalization(A):
    """
    A: [N, N]
    """
    D = torch.pow(A.sum(1).float(), -1)
    # D = torch.pow(A.sum(1).float(), -5)
    D = torch.diag(D)
    adj = torch.matmul(D, A)
    # adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def label_graph(label):
    """
    label: torch.Size([B])
    """
    B = label.size(0)  # B

    # construct semantic graph
    l1 = label.unsqueeze(1).repeat(1, B)  # torch.Size([B, B])
    l2 = label.t().repeat(B, 1)  # torch.Size([B, B])
    adj = l1.eq_(l2)  # torch.Size([B, B])

    # normalization
    adj_loop = adj + torch.eye(B, B)
    A = adj_normalization(adj_loop)  # torch.Size([B, B])
    return A


def lable_graph_mod(labels):
    adj = (labels @ labels.T > 0).float()
    adj = adj + torch.zeros_like(adj).fill_diagonal_(1)
    return adj / adj.sum(dim=1, keepdim=True)


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


if __name__ == "__main__":

    B, C, K = 5, 10, 8
    e, t, l = gen_test_data(B, C, K)

    print(label_graph(t))
    print(lable_graph_mod(l))
    print((label_graph(t) == lable_graph_mod(l)).all())

    # print(euclidean_dist(e, e))
    # print(torch.cdist(e, e))
