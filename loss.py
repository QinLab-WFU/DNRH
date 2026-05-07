import torch
import torch.nn.functional as F
from torch import nn

from Hyp_ViT.hyptorch.pmath import dist_matrix


class DyBGRLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # tau - temperature
        self.tau = 1.0

        # hyp_c - hyperbolic curvature, "0" enables sphere mode
        self.hyp_c = 0.1
        self.dist_f = lambda x, y: -dist_matrix(x, y, c=self.hyp_c)

    def forward(self, batch, labels):
        target = (labels @ labels.T > 0).float()
        logits = self.dist_f(batch, batch)  # Note: use similarity here not distance

        # loss = F.cross_entropy(logits / self.tau, target)  # <- buggy NCA

        logits.fill_diagonal_(torch.finfo(logits.dtype).min)  # no i in Ci: see Eq. 2 in the paper ProxyNCA++
        exp = F.softmax(logits / self.tau, dim=1)
        exp = torch.sum(exp * target, dim=1)  # <- is NCA!
        non_zero = exp != 0
        loss = -torch.log(exp[non_zero]).mean()

        return loss


if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 4, 10, 8
    e, t, l = gen_test_data(B, C, K)
