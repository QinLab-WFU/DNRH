import numpy as np
import torch

# 引入对称的标签
def add_noise_to_labels(labels, noise_rate=0.2, num_classes=None, device=None):
    """
    对多标签标签进行噪声注入：
    随机挑选 noise_rate 比例的样本，
    在这些样本的标签中，随机翻转一个为1的标签为0，同时随机翻转一个为0的标签为1。

    参数：
    labels: Tensor，shape=(batch_size, num_labels)，0/1的多标签矩阵
    noise_rate: 噪声比例，0~1之间
    num_classes: 标签维度，一般等于 labels.shape[1]
    device: torch.device，返回tensor所在设备

    返回：
    噪声后的标签 Tensor，类型float32，和输入设备一致
    """

    labels_np = labels.cpu().numpy()
    num_samples, num_labels = labels_np.shape
    num_noise = int(num_samples * noise_rate)

    noise_indices = np.random.choice(num_samples, num_noise, replace=False)

    for i in noise_indices:
        ones_indices = np.where(labels_np[i, :] == 1)[0]
        zeros_indices = np.where(labels_np[i, :] == 0)[0]

        if len(ones_indices) > 0:
            j = np.random.choice(ones_indices)
            labels_np[i, j] = 0

        if len(zeros_indices) > 0:
            j = np.random.choice(zeros_indices)
            labels_np[i, j] = 1

    if device is None:
        device = labels.device  # 默认使用输入标签的设备

    return torch.tensor(labels_np, dtype=torch.float32, device=device)


import numpy as np

# 引入非对称的标签
def add_asymmetric_noise(labels, noise_rate=0.1, mapping=None, device=None):
    """
    对多标签标签进行非对称噪声注入：
    按 mapping 规则，将一部分样本的标签翻转为特定类别。

    参数：
    labels: Tensor，shape=(batch_size, num_labels)，0/1的多标签矩阵
    noise_rate: 噪声比例，0~1之间
    mapping: dict，非对称噪声映射规则，如 {0: 1, 2: 3} 表示 0类错成1类, 2类错成3类
    device: torch.device

    返回：
    噪声后的标签 Tensor
    """
    labels_np = labels.cpu().numpy()
    num_samples = labels_np.shape[0]
    num_noise = int(num_samples * noise_rate)

    noise_indices = np.random.choice(num_samples, num_noise, replace=False)

    for i in noise_indices:
        for src, tgt in mapping.items():
            if labels_np[i, src] == 1:  # 只翻转特定类别
                labels_np[i, src] = 0
                labels_np[i, tgt] = 1

    if device is None:
        device = labels.device

    return torch.tensor(labels_np, dtype=torch.float32, device=device)


# import numpy as np
# import torch
#
# def add_noise_to_labels(labels):
#     labels = labels.cpu().numpy()  # 转换为 numpy
#     num_samples, num_labels = labels.shape
#     num_noise = int(num_samples * 0.7) #0.1 0.3 0.5
#
#     noise_indices = np.random.choice(num_samples, num_noise, replace=False)
#
#     for i in noise_indices:
#         ones_indices = np.where(labels[i, :] == 1)[0]
#         zeros_indices = np.where(labels[i, :] == 0)[0]
#
#         if len(ones_indices) > 0:
#             j = np.random.choice(ones_indices)
#             labels[i, j] = 0
#
#         if len(zeros_indices) > 0:
#             j = np.random.choice(zeros_indices)
#             labels[i, j] = 1
#
#     return torch.tensor(labels, dtype=torch.float32).to(0)  # 转换回 Tensor 并放回 GPU