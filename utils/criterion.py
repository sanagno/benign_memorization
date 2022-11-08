import torch
import torch.nn as nn


def get_criterion(args):
    if args.use_mse_loss:
        return MyMSELoss()

    return MultiLabelSmoothingCrossEntropyLoss(args)


class MyMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, y):
        return self.loss_fn(x, y.float())


class NonContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x_i, x_j):
        return (x_i - x_j).pow(2).mean(-1).mean()


class MultiLabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, args, dim=-1):
        super().__init__()
        self.total_mass = (
            1
            if (args.persamplerandomlabels_prob_per_class is None)
            else (
                args.persamplerandomlabels_prob_per_class
                * args.persamplerandomlabels_numclasses_per_sample
            )
        )

        self.smoothing = (
            args.label_smoothing if args.label_smoothing is not None else 0.0
        )
        self.cls = args.persamplerandomlabels_numclasses
        self.dim = dim

    def forward(self, pred, target):
        # target is also of size [batch_size, num_classes]
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = target * (1 - self.smoothing)
            true_dist += self.smoothing * self.total_mass / self.cls

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
