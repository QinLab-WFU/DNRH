import torch
import torchvision
from torch import nn

from DyBGR.utils import MaxminNorm
from Hyp_ViT.hyptorch.nn import ToPoincare


def build_model(args, pretrained=True):
    if args.backbone == "resnet50":
        net = ResNet50Mod(
            args.n_bits,
            pretrained,
            graph_mode=args.graph_mode,
            graph_k=args.graph_k,
            graph_t=args.graph_t,
        ).to(args.device)
        return net


class GraphLearningProp(nn.Module):
    def __init__(self, dim, graph_mode="dynamic", K=30, T=3):
        super().__init__()
        self.graph_mode = graph_mode
        self.K = K
        self.T = T
        self.eps = 0.5
        self.lam = 0.3
        self.eta = 0.25

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim)

    def _to_full_graph(self, A_sparse, neighbors, B, device):
        """
        A_sparse: [B, K]
        neighbors: [B, K]
        return: [B, B]
        """
        A_full = torch.zeros(B, B, device=device)
        A_full.scatter_(1, neighbors, A_sparse)
        A_full = 0.5 * (A_full + A_full.T)   # 对称化，便于热图展示
        A_full.fill_diagonal_(0)
        return A_full

    def forward(self, X, L, return_vis=False, force_T=None):

        B, _ = X.shape

        T = self.T if force_T is None else force_T

        # -----------------------------------------------------
        # prevent K overflow
        # -----------------------------------------------------
        K_eff = min(self.K, B - 2)

        if K_eff < 1:
            raise ValueError(
                f"Batch size {B} is too small for graph learning."
            )

        # -----------------------------------------------------
        # save initial feature
        # -----------------------------------------------------
        feat_before = X.clone()

        # -----------------------------------------------------
        # initial relation metric
        # -----------------------------------------------------
        D1 = (
                torch.cdist(X, X)
                - 2 * self.eta * L
                - 0.00001 * MaxminNorm(X @ X.T)
        )

        D2 = MaxminNorm(D1)

        D = self.relu(D2)

        # smaller distance -> closer relation
        Dx, index = torch.sort(D)

        # -----------------------------------------------------
        # gamma
        # -----------------------------------------------------
        dk = Dx[:, K_eff + 1]

        dk_sum = torch.sum(
            Dx[:, 1:K_eff + 1],
            dim=1
        )

        gamma = torch.mean(
            0.5 * (K_eff * dk - dk_sum)
        )

        # -----------------------------------------------------
        # delta
        # -----------------------------------------------------
        delta = 1 / K_eff * (
                1
                + Dx[:, 1:K_eff + 1].sum(
            dim=1,
            keepdim=True
        ) / (2 * gamma + 1e-8)
        )

        # -----------------------------------------------------
        # iterative refinement
        # -----------------------------------------------------
        F = X

        graph_list = []

        for _ in range(T):
            # dynamic attention affinity
            attn = MaxminNorm(F @ F.T)

            # refined distance
            dis = torch.gather(
                D,
                dim=1,
                index=index[:, 1:K_eff + 1]
            ) - self.lam * torch.gather(
                attn,
                dim=1,
                index=index[:, 1:K_eff + 1]
            )

            # adaptive graph weights
            A = self.relu(
                delta.expand(B, K_eff)
                - dis / (2 * gamma + 1e-8)
            )

            neighbors = index[:, 1:K_eff + 1]

            # -------------------------------------------------
            # dense graph for visualization
            # -------------------------------------------------
            A_full = self._to_full_graph(
                A,
                neighbors,
                B,
                X.device
            )

            graph_list.append(
                A_full.detach().cpu()
            )

            # -------------------------------------------------
            # neighborhood aggregation
            # -------------------------------------------------
            AX = (
                    X[neighbors]
                    * A.unsqueeze(-1)
            ).sum(dim=1)

            F = (
                    self.eps * AX
                    + (1 - self.eps) * X
            )

        # -----------------------------------------------------
        # save refined feature
        # -----------------------------------------------------
        feat_after = F.clone()

        # -----------------------------------------------------
        # visualization dict
        # -----------------------------------------------------
        if return_vis:
            init_graph = 1.0 - D

            init_graph = 0.5 * (
                    init_graph + init_graph.T
            )

            init_graph.fill_diagonal_(0)

            vis_dict = {

                "init_graph":
                    init_graph.detach().cpu(),

                "one_shot_graph":
                    graph_list[0],

                "final_graph":
                    graph_list[-1],

                "all_graphs":
                    graph_list,

                # NEW
                "init_features":
                    feat_before.detach().cpu(),

                "final_features":
                    feat_after.detach().cpu(),
            }

            return F, vis_dict

        return F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, graph_mode="dynamic", K=30, T=3, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.glp = GraphLearningProp(dim, graph_mode=graph_mode, K=K, T=T)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, labels, return_vis=False, force_T=None):

        if return_vis:

            glp_feat, vis_dict = self.glp(
                self.norm1(x),
                self.label_graph(labels),
                return_vis=True,
                force_T=force_T,
            )

            x = x + glp_feat

            x = x + self.mlp(self.norm2(x))

            return x, vis_dict

        else:

            x = x + self.glp(
                self.norm1(x),
                self.label_graph(labels)
            )

            x = x + self.mlp(self.norm2(x))

            return x

    def label_graph(self, labels):
        """
        construct semantic graph
        """
        adj = (labels @ labels.T > 0).float()
        adj = adj + torch.zeros_like(adj).fill_diagonal_(1)
        return adj / adj.sum(dim=1, keepdim=True)


class ResNet50Mod(nn.Module):
    def __init__(self, n_bits, pretrained=True, graph_mode="dynamic", graph_k=30, graph_t=3):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet50(weights=weights)
        n_channels = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.depth = 2
        self.blocks = nn.ModuleList([
            Block(dim=n_channels, graph_mode=graph_mode, K=graph_k, T=graph_t, mlp_ratio=4, drop=0.0)
            for _ in range(self.depth)
        ])

        last = ToPoincare(
            c=0.1,
            ball_dim=n_bits,
            riemannian=False,
            clip_r=2.3,
        )
        self.head = nn.Sequential(nn.Linear(n_channels, n_bits), last)

    def forward(self, x, y=None):
        x = self.backbone(x)
        pred_x = x

        if y is not None:
            for layer in self.blocks:
                x = layer(x, y)
            x = torch.cat([pred_x, x], 0)

        x = self.head(x)
        return x

    @torch.no_grad()
    def extract_graphs(self, x, y, block_idx=0, force_T=None):
        """
        Extract visualization graphs from a specific DyBGR block.

        Args:
            x: input images, shape [B, 3, H, W]
            y: labels, shape [B, C]
            block_idx: which block to visualize
            force_T: override the default iteration number

        Returns:
            vis_dict containing:
                init_graph, one_shot_graph, final_graph, all_graphs
        """
        self.eval()

        feat = self.backbone(x)
        block = self.blocks[block_idx]

        _, vis_dict = block(
            feat,
            y,
            return_vis=True,
            force_T=force_T,
        )
        return vis_dict


if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 50, 10, 8
    e, t, l = gen_test_data(B, C, K)
    # net = Block(K)
    # z = net(e, l)
    # print(z.shape)

    net = ResNet50Mod(K)
    # net.eval()
    x = torch.randn(B, 3, 224, 224)
    z = net(x, l)
    print(z.shape)
