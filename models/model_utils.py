import math
import torch
from torch import nn


def transformation(covs, engine="symeig"):
    # covs = covs.to(linalg_device)
    if engine == "cholesky":
        C = torch.cholesky(covs)
        W = torch.triangular_solve(
            torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False
        )[0].transpose(1, 2)
    else:
        if engine == "symeig":
            # S, U = torch.linalg.eigh(covs, UPLO="U")
            S, U = torch.symeig(covs, eigenvectors=True, upper=True)
        elif engine == "svd":
            U, S, _ = torch.svd(covs)
        elif engine == "svd_lowrank":
            U, S, _ = torch.svd_lowrank(covs)
        elif engine == "pca_lowrank":
            U, S, _ = torch.pca_lowrank(covs, center=False)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1, 2))
    return W


class ShuffledGroupWhitening(nn.Module):
    def __init__(self, num_features, num_groups=None, shuffle=True, engine="symeig"):
        super(ShuffledGroupWhitening, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups is not None:
            assert self.num_features % self.num_groups == 0

        print("Setting ShuffledGroupWhitening num_groups to {}".format(self.num_groups))

        # self.momentum = momentum
        self.register_buffer("running_mean", None)
        self.register_buffer("running_covariance", None)
        self.shuffle = shuffle if self.num_groups != 1 else False
        self.engine = engine

    def forward(self, x):
        N, D = x.shape
        if self.num_groups is None:
            G = math.ceil(
                2 * D / N
            )  # automatic, the grouped dimension 'D/G' should be half of the batch size N
            # print(G, D, N)
        else:
            G = self.num_groups
        if self.shuffle:
            new_idx = torch.randperm(D)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D // G)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0, 1)  # G, N, D//G
        # covs = x.transpose(1,2).bmm(x) / (N-1) #  G, D//G, N @ G, N, D//G -> G, D//G, D//G
        covs = (
            x.transpose(1, 2).bmm(x) / N
            + 1e-4 * torch.eye(x.shape[-1]).to(x.device)[None, ...]
        )
        W = transformation(covs, engine=self.engine)
        x = x.bmm(W)
        if self.shuffle:
            return x.transpose(1, 2).flatten(0, 1)[torch.argsort(new_idx)].t()
        else:
            return x.transpose(0, 1).flatten(1)


class MyIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class OurBatchNorm(nn.Module):
    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x):
        if len(x.shape) == 2:
            out = (x - self.running_mean[None, :]) / (self.running_var[None, :] + 1e-7)
        else:
            out = (x - self.running_mean[None, :, None, None]) / (
                self.running_var[None, :, None, None] + 1e-7
            )

        if self.training:
            if len(x.shape) == 2:
                mean = x.mean(0).detach()
                var = x.std(0, unbiased=False).detach()
            else:
                mean = x.mean([0, 2, 3]).detach()
                var = x.std([0, 2, 3], unbiased=False).detach()

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

        return out


def get_normalization(name, dim=None, partial=False, *args, **kwargs):
    if name == "none":
        if partial:
            return lambda x: MyIdentity()
        return MyIdentity()
    elif name == "bn1d":
        if partial:
            return lambda dim: nn.BatchNorm1d(dim, affine=kwargs["affine"])
        return nn.BatchNorm1d(dim, affine=kwargs["affine"])
    elif name == "bn2d":
        if partial:
            return lambda dim: nn.BatchNorm2d(dim, affine=kwargs["affine"])
        return nn.BatchNorm2d(dim, affine=kwargs["affine"])
    elif name == "ourbn":
        if partial:
            return lambda dim: OurBatchNorm(dim)
        return OurBatchNorm(dim)
    elif name == "ln":
        if partial:
            return lambda dim: nn.LayerNorm(dim)
        return nn.LayerNorm(dim)
    elif name == "gn":
        if partial:
            return lambda dim: nn.GroupNorm(dim // 8, dim)
        return nn.GroupNorm(dim // 8, dim)
    elif name == "zca":
        if partial:
            return lambda dim: ShuffledGroupWhitening(
                num_groups=dim // 8, num_features=dim
            )
        return ShuffledGroupWhitening(num_groups=dim // 8, num_features=dim)
    else:
        raise ValueError("Unknown normalization: {}".format(name))
