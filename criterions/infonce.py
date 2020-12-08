import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature, normalize):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, prediction_embs, target_embs, choices_dim=1, output_head=None, target_head=None):
        assert len(prediction_embs.shape) == len(target_embs.shape)
        assert all([s1==s2 for s1, s2 in zip(prediction_embs.shape, target_embs.shape)])
        if output_head is not None and target_head is None:
            all_embs = torch.cat((prediction_embs, target_embs), 0)
            all_embs = output_head(all_embs)
            prediction_embs, target_embs = torch.chunk(all_embs, chunks=2, dim=0)
        elif output_head is not None and target_head is not None:
            prediction_embs = output_head(prediction_embs)
            target_embs = target_head(target_embs)

        if self.normalize:
            prediction_embs = F.normalize(prediction_embs, p=2, dim=-1)
            target_embs = F.normalize(target_embs, p=2, dim=-1)

        if choices_dim < 0 :
            choices_dim += len(target_embs.shape)
        if choices_dim != len(target_embs.shape)-2:
            target_embs = target_embs.transpose(choices_dim, len(target_embs.shape)-2)
            prediction_embs = prediction_embs.transpose(choices_dim, len(target_embs.shape)-2)
        if len(target_embs.shape) != 3:
            sz = prediction_embs.shape
            prediction_embs = prediction_embs.view(-1, sz[-2], sz[-1])
            target_embs = target_embs.view(-1, sz[-2], sz[-1])

        # Compute scores
        scores = torch.bmm(prediction_embs, target_embs.transpose(-2, -1)) / self.temperature
        scores = scores.flatten(0, 1)

        # Labels
        bs, n_choices = target_embs.shape[:2]
        labels = torch.arange(n_choices, device=prediction_embs.device)
        if bs > 1:
            labels = torch.stack([labels] * bs, 0).flatten(0, 1)

        # Compute loss
        loss = F.cross_entropy(scores, labels)
        return loss, scores

