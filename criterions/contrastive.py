import torch
from torch import nn
import torch.nn.functional as F
from criterions.infonce import InfoNCELoss

__all__ = ['ContrastiveLoss', 'HardContrastiveLoss']


class Head(nn.Module):
    def __init__(self, input_dim, proj_dims):
        super(Head, self).__init__()
        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        projection = []
        for i, d in enumerate(proj_dims):
            projection += [nn.Linear(input_dim, d)]
            input_dim = d
            if i < len(proj_dims)-1:
                projection += [nn.ReLU(inplace=True)]
        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class ContrastiveLoss(nn.Module):
    def __init__(self,
                 input_dim=512,
                 proj_dim=None,
                 target='cross-modal',
                 temperature=0.07,
                 normalize=True):
        super(ContrastiveLoss, self).__init__()
        self.video_projection = Head(input_dim, proj_dim) if proj_dim is not None else None
        self.audio_projection = Head(input_dim, proj_dim) if proj_dim is not None else None
        assert target in {'cross-modal', 'within-modal'}
        self.target = target
        assert temperature > 0.
        self.temperature = temperature
        assert isinstance(normalize, bool)
        self.normalize = normalize
        self.contrastive_loss = InfoNCELoss(temperature=temperature, normalize=normalize)

    def predict(self, video_emb, audio_emb):
        video_emb = self.video_projection(video_emb)
        audio_emb = self.video_projection(audio_emb)
        return video_emb, audio_emb

    def forward(self, video_emb, audio_emb, *args):
        losses, scores = {}, {}
        if self.target == 'cross-modal':
            losses['V2A'], scores['V2A'] = self.contrastive_loss(
                video_emb, audio_emb, choices_dim=0, output_head=self.video_projection, target_head=self.audio_projection)
            losses['A2V'], scores['A2V'] = self.contrastive_loss(
                audio_emb, video_emb, choices_dim=0, output_head=self.audio_projection, target_head=self.video_projection)
        elif self.target == 'within-modal':
            losses['V2V'], scores['V2V'] = self.contrastive_loss(
                audio_emb, audio_emb, choices_dim=0, output_head=self.audio_projection)
            losses['A2A'], scores['A2A'] = self.contrastive_loss(
                audio_emb, video_emb, choices_dim=0, output_head=self.video_projection)

        total_loss = sum([losses[k] for k in losses]) / float(len(losses))

        # Compute scores for tensorboard
        with torch.no_grad():
            metrics = {}
            for k in scores:
                n_preds, n_choices = scores[k].shape[0] // scores[k].shape[1], scores[k].shape[1]
                labels = torch.arange(n_choices, device=video_emb.device)
                labels = torch.stack([labels] * n_preds, 0).flatten()
                scores_pos = scores[k][torch.arange(0, n_preds*n_choices), labels]
                scores_neg = (scores[k].sum() - scores_pos.sum()) / float(scores[k].numel()-scores_pos.numel())
                metrics[f"Scores/{k}/Pos"] = scores_pos.mean()
                metrics[f"Scores/{k}/Neg"] = scores_neg
                metrics[f"Loss/{k}"] = losses[k]

        return total_loss, metrics


class HardContrastiveLoss(nn.Module):
    def __init__(self,
                 easy_coeff=1.,
                 hard_coeff=1.,
                 temperature=0.07,
                 normalize=True):
        super(HardContrastiveLoss, self).__init__()
        self.easy_coeff = easy_coeff
        self.hard_coeff = hard_coeff
        assert temperature > 0.
        self.temperature = temperature
        assert isinstance(normalize, bool)
        self.normalize = normalize

    def forward(self, video_emb, audio_emb):
        bs, n_aug, dim = video_emb.shape
        assert n_aug == 2, f'Within-modal contrastive loss requires 2 augmentation of each sample. {n_aug} provided.'

        # Normalize embeddings
        if self.normalize:
            video_emb = nn.functional.normalize(video_emb, p=2, dim=2)
            audio_emb = nn.functional.normalize(audio_emb, p=2, dim=2)

        # Compute similarities
        targets = torch.arange(video_emb.size(0), device=video_emb.device)
        scores_easy = video_emb[:, 0].mm(audio_emb[:, 0].T) / self.temperature
        scores_hard_v2a = video_emb[:, 0].mm(audio_emb[:, 1].T) / self.temperature
        scores_hard_v2a = torch.cat([scores_easy, scores_hard_v2a], 1)
        scores_hard_a2v = audio_emb[:, 0].mm(video_emb[:, 1].T) / self.temperature
        scores_hard_a2v = torch.cat([scores_easy.T, scores_hard_a2v], 1)
        scores = {'V2A-Easy': scores_easy, 'A2V-Easy': scores_easy.T,
                  'V2A-Hard': scores_hard_v2a, 'A2V-Hard': scores_hard_a2v}

        # Compute losses
        loss = {k: F.cross_entropy(scores[k], targets) for k in scores}
        loss_easy = (loss['V2A-Easy'] + loss['A2V-Easy']) / 2.
        loss_hard = (loss['V2A-Hard'] + loss['A2V-Hard']) / 2.
        total_loss = loss_easy * self.easy_coeff + loss_hard * self.hard_coeff

        # Log to tensorboard
        with torch.no_grad():
            metrics = {
                "Scores/Pos": torch.diag(scores_easy).mean(),
                "Scores/HardNeg-V2A": torch.diag(scores_hard_v2a[:, bs:]).mean(),
                "Scores/HardNeg-A2V": torch.diag(scores_hard_a2v[:, bs:]).mean(),
                "Scores/EasyNeg": (scores_easy.sum() - torch.diag(scores_easy).sum()) / (bs*(bs-1))
            }
            for k in scores:
                metrics[f"Loss/{k}"] = loss[k]

        return total_loss, metrics
