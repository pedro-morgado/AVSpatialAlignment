import torch
from torch import nn
from torch.nn import functional as F
from criterions.transformers import SimpleTransformer, NoCtxTransformer
from criterions.infonce import InfoNCELoss
from criterions.contrastive import Head

__all__ = ['Video2AudioSeq', 'Video2AudioNoCtx']


DEFAULT_TRANSFORMER_ARGS = {
    'depth': 2,
    'model_dim': 512,
    'expansion': 4,
    'attention_heads': 8,
    'attention_type': 'self',
    'dropout': 0.1,
}

DEFAULT_CONTRASTIVE_ARGS = {
    'normalize': True,
    'temperature': 0.07,
}


class Video2AudioSeq(nn.Module):
    def __init__(self,
                 use_seq2seq_transformer=True,
                 use_context=True,
                 contrast='avc',
                 contrastive_loss_args=DEFAULT_CONTRASTIVE_ARGS,
                 transformer_args=DEFAULT_TRANSFORMER_ARGS,
                 checkpoint=None):
        super(Video2AudioSeq, self).__init__()

        if use_context:
            transformer_args['attention_type'] = 'self'
            self.v2a_transformer = SimpleTransformer(**transformer_args) if use_seq2seq_transformer else None
            self.a2v_transformer = SimpleTransformer(**transformer_args) if use_seq2seq_transformer else None
            if checkpoint is not None:
                ckp = torch.load(checkpoint, map_location='cpu')['train_criterion']
                ckp_v2a_transf = {k.replace('losses.0.v2a_transformer.', ''): ckp[k] for k in ckp if 'v2a_transformer' in k}
                self.v2a_transformer.load_state_dict(ckp_v2a_transf)
                ckp_a2v_transf = {k.replace('losses.0.a2v_transformer.', ''): ckp[k] for k in ckp if 'a2v_transformer' in k}
                self.a2v_transformer.load_state_dict(ckp_a2v_transf)
        else:
            self.v2a_transformer = NoCtxTransformer(**transformer_args) if use_seq2seq_transformer else None
            self.a2v_transformer = NoCtxTransformer(**transformer_args) if use_seq2seq_transformer else None

        self.video_head = Head(512, 128)
        self.audio_head = Head(512, 128)
        self.video_pred_head = Head(512, 128)
        self.audio_pred_head = Head(512, 128)

        assert contrast in {'avc', 'hard', 'easy+hard'}
        self.contrast = contrast
        self.contrastive_loss = InfoNCELoss(**contrastive_loss_args)

    def contrastive_joint_loss(self, x_pred, x_trg):
        bs, naug = x_pred.shape[:2]
        x_pred = F.normalize(x_pred, p=2, dim=-1)
        x_trg = F.normalize(x_trg, p=2, dim=-1)
        x_pred = x_pred.reshape(-1, x_pred.shape[-1])
        x_trg = x_trg.reshape(-1, x_trg.shape[-1])

        pool_idx = torch.zeros((bs, naug, bs + naug - 1), dtype=torch.long)
        label = torch.zeros((bs, naug), dtype=torch.long)
        idx = torch.arange(bs * naug).view(bs, naug)
        for i in range(bs):
            for j in range(naug):
                pool_idx[i, j] = torch.cat((idx[i, :], idx[:i, j], idx[i + 1:, j]))
                label[i, j] = j
        pool_idx = pool_idx.flatten(0, 1)
        label = label.flatten(0, 1)
        scores = torch.bmm(x_trg[pool_idx], x_pred.unsqueeze(2)).squeeze(-1) / self.contrastive_loss.temperature
        loss = F.cross_entropy(scores, label.to(scores.device))
        with torch.no_grad():
            avc_acc = torch.all(scores.gather(1, label.unsqueeze(1).to(scores.device)) > scores[:, naug:], 1).float().mean() * 100
            avsa_acc = torch.all(scores.gather(1, label.unsqueeze(1).to(scores.device)) >= scores[:, :naug], 1).float().mean() * 100
            metrics = {'AVC-Acc': avc_acc, 'AVSA-Acc': avsa_acc}
        return loss, metrics
    
    def predict(self, video_emb, audio_emb, proj=False):
        audio_emb_pred = self.v2a_transformer(video_emb) if self.v2a_transformer is not None else video_emb
        video_emb_pred = self.a2v_transformer(audio_emb) if self.a2v_transformer is not None else audio_emb
        if proj:
            bs, n_aug = video_emb.shape[:2]
            if self.video_head is not None:
                video_emb = self.video_head(video_emb.flatten(0, 1)).view(bs, n_aug, -1)
                video_emb_pred = self.video_pred_head(video_emb_pred.flatten(0, 1)).view(bs, n_aug, -1)
            if self.audio_head is not None:
                audio_emb = self.audio_head(audio_emb.flatten(0, 1)).view(bs, n_aug, -1)
                audio_emb_pred = self.audio_pred_head(audio_emb_pred.flatten(0, 1)).view(bs, n_aug, -1)
            return video_emb, audio_emb, video_emb_pred, audio_emb_pred
        return audio_emb_pred, video_emb_pred

    def forward(self, video_emb, audio_emb, positions):
        audio_emb_pred = self.v2a_transformer(video_emb) if self.v2a_transformer is not None else video_emb
        video_emb_pred = self.a2v_transformer(audio_emb) if self.a2v_transformer is not None else audio_emb

        losses, scores = {}, {}
        if self.contrast == 'avc':
            video_emb_pred = torch.max(video_emb_pred, 1)[0]
            audio_emb_pred = torch.max(audio_emb_pred, 1)[0]
            video_emb = torch.max(video_emb, 1)[0]
            audio_emb = torch.max(audio_emb, 1)[0]
            losses[f'A2V'], scores[f'A2V'] = self.contrastive_loss(video_emb_pred, video_emb, 0, output_head=self.video_pred_head, target_head=self.video_head)
            losses[f'V2A'], scores[f'V2A'] = self.contrastive_loss(audio_emb_pred, audio_emb, 0, output_head=self.audio_pred_head, target_head=self.audio_head)
        elif self.contrast == 'hard':
            losses[f'A2V'], scores[f'A2V'] = self.contrastive_loss(video_emb_pred, video_emb, 1, output_head=self.video_pred_head, target_head=self.video_head)
            losses[f'V2A'], scores[f'V2A'] = self.contrastive_loss(audio_emb_pred, audio_emb, 1, output_head=self.audio_pred_head, target_head=self.audio_head)
        else:
            losses[f'A2V'], scores[f'A2V'] = self.contrastive_joint_loss(self.video_pred_head(video_emb_pred), self.video_head(video_emb))
            losses[f'V2A'], scores[f'V2A'] = self.contrastive_joint_loss(self.audio_pred_head(audio_emb_pred), self.audio_head(audio_emb))

        total_loss = sum([losses[k] for k in losses]) / len(losses)

        # Compute metrics for tensorboard
        with torch.no_grad():
            metrics = {}
            for k in scores:
                metrics[f"Loss/{k}"] = losses[k]
                for k2 in scores[k]:
                    metrics[f"Metrics/{k}/{k2}"] = scores[k][k2]

        return total_loss, metrics


class Video2AudioNoCtx(nn.Module):
    def __init__(self,
                 use_seq2seq_transformer=True,
                 contrast='easy',
                 contrastive_loss_args=DEFAULT_CONTRASTIVE_ARGS,
                 transformer_args=DEFAULT_TRANSFORMER_ARGS):
        super(Video2AudioNoCtx, self).__init__()

        self.v2a_transformer = NoCtxTransformer(**transformer_args) if use_seq2seq_transformer else None
        self.a2v_transformer = NoCtxTransformer(**transformer_args) if use_seq2seq_transformer else None

        self.video_head = Head(512, 128)
        self.audio_head = Head(512, 128)

        assert contrast in {'avc', 'hard', 'easy+hard'}
        self.contrast = contrast
        self.contrastive_loss = InfoNCELoss(**contrastive_loss_args)

    def cross_modal_prediction(self, orig_emb, transformer):
        return transformer(orig_emb) if transformer is not None else orig_emb

    def forward(self, video_emb, audio_emb, positions):
        audio_emb_pred = self.v2a_transformer(video_emb) if self.v2a_transformer is not None else video_emb
        video_emb_pred = self.a2v_transformer(audio_emb) if self.a2v_transformer is not None else audio_emb

        losses, scores = {}, {}
        if self.contrast == 'avc':
            video_emb_pred = torch.max(video_emb_pred, 1)[0]
            audio_emb_pred = torch.max(audio_emb_pred, 1)[0]
            video_emb = torch.max(video_emb, 1)[0]
            audio_emb = torch.max(audio_emb, 1)[0]
            losses[f'A2V'], scores[f'A2V'] = self.contrastive_loss(video_emb_pred, video_emb, 0, output_head=self.video_head)
            losses[f'V2A'], scores[f'V2A'] = self.contrastive_loss(audio_emb_pred, audio_emb, 0, output_head=self.audio_head)
        elif self.contrast == 'hard':
            losses[f'A2V'], scores[f'A2V'] = self.contrastive_loss(video_emb_pred, video_emb, 1, output_head=self.video_head)
            losses[f'V2A'], scores[f'V2A'] = self.contrastive_loss(audio_emb_pred, audio_emb, 1, output_head=self.audio_head)
        else:
            video_emb = video_emb.flatten(0, 1)
            audio_emb = audio_emb.flatten(0, 1)
            video_emb_pred = video_emb_pred.flatten(0, 1)
            audio_emb_pred = audio_emb_pred.flatten(0, 1)

            losses[f'A2V'], scores[f'A2V'] = self.contrastive_loss(video_emb_pred, video_emb, 0, output_head=self.video_head)
            losses[f'V2A'], scores[f'V2A'] = self.contrastive_loss(audio_emb_pred, audio_emb, 0, output_head=self.audio_head)

        total_loss = sum([losses[k] for k in losses]) / len(losses)

        # Compute metrics for tensorboard
        with torch.no_grad():
            metrics = {}
            for k in scores:
                bs, n_choices = scores[k].shape[-2]//scores[k].shape[-1], scores[k].shape[-1]
                labels = torch.arange(n_choices, device=video_emb.device)
                labels = torch.stack([labels] * bs, 0).flatten()
                scores_pos = scores[k][torch.arange(0, bs*n_choices), labels]
                scores_neg = (scores[k].sum() - scores_pos.sum()) / float(scores[k].numel()-scores_pos.numel())
                metrics[f"Scores/{k}/Pos"] = scores_pos.mean()
                metrics[f"Scores/{k}/Neg"] = scores_neg
                metrics[f"Loss/{k}"] = losses[k]

        return total_loss, metrics