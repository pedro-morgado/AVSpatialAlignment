from torch import nn
import criterions

__all__ = ['MultiTask']


class MultiTask(nn.Module):
    def __init__(self, losses):
        super(MultiTask, self).__init__()

        self.losses, self.coeff, self.names = [], [], []
        for loss in losses:
            self.losses += [criterions.__dict__[loss["name"]](**loss["args"])]
            self.coeff += [loss["coeff"]]
            self.names += [loss["name"]]
        self.losses = nn.ModuleList(self.losses)
        self.coeff = [c/sum(self.coeff) for c in self.coeff]

    def forward(self, *x):
        metrics = {}
        total_loss = 0.
        for l, c, n in zip(self.losses, self.coeff, self.names):
            loss, metrics_l = l(*x)
            total_loss += loss * c
            for k in metrics_l:
                metrics[f"{n}/{k}"] = metrics_l[k]
        return total_loss, metrics

