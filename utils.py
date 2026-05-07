import torch


def MaxminNorm(x):
    # 沿着行的方向计算最小值和最大值
    min_vals, _ = torch.min(x, dim=1, keepdim=True)
    max_vals, _ = torch.max(x, dim=1, keepdim=True)

    # 最小-最大缩放，将x的范围缩放到[0, 1]
    scaled_x = (x - min_vals) / (max_vals - min_vals)
    return scaled_x
