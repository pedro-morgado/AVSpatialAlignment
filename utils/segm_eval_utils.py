import os
import yaml
from utils import main_utils
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

def prepare_environment(args, cfg):
    model_cfg = yaml.safe_load(open(args.model_cfg))['model']
    criterion_cfg = yaml.safe_load(open(args.model_cfg))['loss']
    if args.checkpoint_dir is None:
        eval_dir = '{}/{}/eval-{}/fold-{:02d}'.format(model_cfg['model_dir'], model_cfg['name'], cfg['benchmark'], cfg['dataset']['fold'])
    else:
        eval_dir = '{}/eval-{}/fold-{:02d}'.format(args.checkpoint_dir, cfg['benchmark'], cfg['dataset']['fold'])
    os.makedirs(eval_dir, exist_ok=True)
    yaml.safe_dump(cfg, open('{}/config.yaml'.format(eval_dir), 'w'))

    logger = main_utils.Logger(quiet=args.quiet, log_fn='{}/eval.log'.format(eval_dir), rank=args.gpu)
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))

    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  ' + ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))

    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    print_dict(cfg)
    logger.add_line("=" * 30 + "   Model Config   " + "=" * 30)
    print_dict(model_cfg)

    return model_cfg, criterion_cfg, eval_dir, logger


def build_model(feat_cfg, criterion_cfg, cfg, eval_dir, args, logger=None):
    import models
    pretrained_net = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])
    if not cfg['scratch']:
        checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
        ckp = torch.load(checkpoint_fn, map_location='cpu')
        try:
            pretrained_net.load_state_dict({k.replace('module.', ''): ckp['state_dict'][k] for k in ckp['state_dict']})
        except KeyError:
            pretrained_net.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})

    if 'checkpoint' in cfg:
        ckp = torch.load(cfg['checkpoint'], map_location='cpu')
        pretrained_net.video_model.load_state_dict({k.replace('module.model.', ''): ckp['model'][k] for k in ckp['model'] if 'classifier' not in k})

    if cfg['name'] == 'fpn':
        model = FPNWrapper(pretrained_net.video_model, pretrained_net.audio_model, **cfg['args'])
        head_params = [p for n, p in model.named_parameters() if 'video_model' not in n and 'audio_model' not in n]
    elif cfg['name'] == 'fpn-ctx':
        import criterions
        criterion = criterions.__dict__[criterion_cfg['name']](**criterion_cfg['args'])
        if not cfg['scratch']:
            checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
            ckp = torch.load(checkpoint_fn, map_location='cpu')
            criterion.load_state_dict(ckp['train_criterion'])
        v2a_transformer = criterion.losses[0].v2a_transformer if cfg['args']['use_context'] else None
        a2v_transformer = criterion.losses[0].a2v_transformer if cfg['args']['use_context'] else None
        model = FPNCtxWrapper(pretrained_net.video_model, pretrained_net.audio_model, v2a_transformer, a2v_transformer, **cfg['args'])
        head_params = [p for n, p in model.named_parameters() if
                       'video_model' not in n and
                       'audio_model' not in n and
                       'v2a_transformer' not in n and
                       'a2v_transformer' not in n
                       ]
        # head_params = model.fpn.predictor.parameters()
    else:
        raise ValueError
    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    model = model.cuda(args.gpu)
    return model, head_params, ckp_manager


def log_model(model, logger=None):
    if logger is not None:
        logger.add_line("=" * 30 + "   Model   " + "=" * 30)
        logger.add_line(str(model))
        logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
        logger.add_line(main_utils.parameter_description(model))


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, scheduler, epoch, eval_metric=0.):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best=is_best, model_dir=self.checkpoint_dir)

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer=None, scheduler=None, restore_last=False, restore_best=False, logger=None):
        start_epoch = 0
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        if self.checkpoint_exists(last=restore_last, best=restore_best):
            ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
            start_epoch = ckp['epoch']
            model.load_state_dict(ckp['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(ckp['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(ckp['scheduler'])
            if logger is not None:
                logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(checkpoint_fn, start_epoch))
        else:
            if logger is not None:
                logger.add_line("No checkpoint found at '{}'.".format(checkpoint_fn))

        return start_epoch


class TimePool(torch.nn.Module):
    def __init__(self, window=8):
        super(TimePool, self).__init__()
        self.weight = nn.Parameter(torch.ones((1, 1, window, 1, 1), requires_grad=True))
        self.weight.data[:] = 1./window

    def forward(self, x):
        return (x * self.weight).sum(2)


class FPNHead(torch.nn.Module):
    def __init__(self, num_classes, fpn_dim=256, segm_dim=128, inpt_dim=512, cond_dim=0):
        super(FPNHead, self).__init__()
        # Parametric temporal pooling
        self.time_pool5 = TimePool(1)
        self.time_pool4 = TimePool(2)
        self.time_pool3 = TimePool(4)
        self.time_pool2 = TimePool(8)

        # FPN lateral layers
        enc_dim = inpt_dim + cond_dim
        self.lat_fpn_5 = nn.Conv2d(enc_dim, fpn_dim, kernel_size=1, stride=1, padding=0)
        self.lat_fpn_4 = nn.Conv2d(256, fpn_dim, kernel_size=1, stride=1, padding=0)
        self.lat_fpn_3 = nn.Conv2d(128, fpn_dim, kernel_size=1, stride=1, padding=0)
        self.lat_fpn_2 = nn.Conv2d(64,  fpn_dim, kernel_size=1, stride=1, padding=0)

        # FPN output layers
        self.out_fpn_5 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.out_fpn_4 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.out_fpn_3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.out_fpn_2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)

        # Semantic branch
        self.upsampling5 = nn.Sequential(
            nn.Conv2d(fpn_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(segm_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(segm_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.upsampling4 = nn.Sequential(
            nn.Conv2d(fpn_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(segm_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.upsampling3 = nn.Sequential(
            nn.Conv2d(fpn_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.upsampling2 = nn.Sequential(
            nn.Conv2d(fpn_dim, segm_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, segm_dim),
            nn.ReLU(inplace=True),
        )
        self.predictor = nn.Conv2d(segm_dim, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'Conv2D_Small':
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def forward(self, inpt_embs, cond_embs=None):
        # Bottom-up using backbone
        c2, c3, c4, c5 = [inpt_embs[k] for k in ['conv2x', 'conv3x', 'conv4x', 'conv5x']]
        if cond_embs is not None:
            bs, nf = cond_embs.shape
            cond_embs = cond_embs.view(bs, nf, 1, 1, 1)
            c5 = torch.cat((c5, cond_embs.repeat(1, 1, *c5.shape[2:])), 1)

        # Top-down FPN
        i5 = self.lat_fpn_5(self.time_pool5(c5))
        i4 = self.lat_fpn_4(self.time_pool4(c4)) + F.interpolate(i5, scale_factor=2, mode='nearest')
        i3 = self.lat_fpn_3(self.time_pool3(c3)) + F.interpolate(i4, scale_factor=2, mode='nearest')
        i2 = self.lat_fpn_2(self.time_pool2(c2)) + F.interpolate(i3, scale_factor=2, mode='nearest')

        # FPN output
        p5 = self.out_fpn_5(i5)
        p4 = self.out_fpn_4(i4)
        p3 = self.out_fpn_3(i3)
        p2 = self.out_fpn_2(i2)

        # Semantic
        s5 = self.upsampling5(p5)
        s4 = self.upsampling4(p4)
        s3 = self.upsampling3(p3)
        s2 = self.upsampling2(p2)
        s = self.predictor(s5 + s4 + s3 + s2)
        return s


class FPNWrapper(torch.nn.Module):
    def __init__(self, video_model, audio_model, num_classes, fpn_dim=256, segm_dim=128, video_only=False):
        super(FPNWrapper, self).__init__()
        self.video_only = video_only

        self.video_model = video_model
        if not video_only:
            self.audio_model = audio_model

        self.fpn = FPNHead(num_classes, fpn_dim=fpn_dim, segm_dim=segm_dim, inpt_dim=512, cond_dim=0 if video_only else 512)

    def forward(self, video, audio):
        video_embs = self.video_model(video, return_embs=True)
        audio_embs = self.audio_model(audio) if not self.video_only else None
        s = self.fpn(video_embs, audio_embs)
        return s


class FPNCtxWrapper(torch.nn.Module):
    def __init__(self, video_model, audio_model, v2a_transformer, a2v_transformer, num_classes, fpn_dim=256, segm_dim=128, use_audio=True, use_context=True):
        super(FPNCtxWrapper, self).__init__()
        self.use_audio = use_audio
        self.use_context = use_context

        self.video_model = video_model
        inpt_dim = video_model.out_dim
        cond_dim = 0
        if use_audio:
            self.audio_model = audio_model
            cond_dim += self.audio_model.out_dim
        if use_context:
            self.v2a_transformer = v2a_transformer
            cond_dim += self.video_model.out_dim
            if use_audio:
                self.a2v_transformer = a2v_transformer
                cond_dim += self.audio_model.out_dim

        self.fpn = FPNHead(num_classes, fpn_dim=fpn_dim, segm_dim=segm_dim, inpt_dim=inpt_dim, cond_dim=cond_dim)

    def forward(self, video, audio):
        bs, naug = video.shape[:2]
        video_embs = self.video_model(video.flatten(0, 1), return_embs=True)
        cond_embs = []
        if self.use_context:
            video_out = video_embs['pool'].reshape(bs, naug, self.video_model.out_dim)
            video_ctx = self.v2a_transformer(video_out)
            cond_embs += [video_ctx.flatten(0, 1)]
        if self.use_audio:
            audio_embs = self.audio_model(audio.flatten(0, 1))
            cond_embs += [audio_embs.flatten(1, -1)]
            if self.use_context:
                audio_out = audio_embs.reshape(bs, naug, self.audio_model.out_dim)
                audio_ctx = self.a2v_transformer(audio_out)
                cond_embs += [audio_ctx.flatten(0, 1)]
        cond_embs = torch.cat(cond_embs, 1) if cond_embs else None
        s = self.fpn(video_embs, cond_embs)
        return s.reshape(bs, naug, *s.shape[1:])

    def freeze_backbone(self):
        for p in self.video_model.parameters():
            p.requires_grad = False
        if self.use_audio:
            for p in self.audio_model.parameters():
                p.requires_grad = False
        if self.use_context:
            for p in self.v2a_transformer.parameters():
                p.requires_grad = False
        if self.use_audio and self.use_context:
            for p in self.a2v_transformer.parameters():
                p.requires_grad = False

    def head_parameters(self):
        return self.fpn.parameters()


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, *inputs):
        outs = []
        for i in range(0, inputs[0].shape[0], self.batch_size):
            inputs = [x[i:i+self.batch_size] for x in inputs]
            outs += [self.model(*inputs)]
        return torch.cat(outs, 0)


def distribute_model_to_cuda(model, args, cfg):
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    return model


def build_dataloader(db_cfg, split_cfg, num_workers, distributed):
    import numpy as np
    import torch.utils.data as data
    from datasets import preprocessing

    joint_transform = preprocessing.VideoLabelPrep_Segm(
        resize=(db_cfg['resize'], db_cfg['resize']),
        crop=(db_cfg['crop_size'], db_cfg['crop_size']),
        augment=split_cfg['use_augmentation'],
    )
    video_transform = preprocessing.VideoPrep_Segm(
        num_frames=int(db_cfg['clip_duration']),
        pad_missing=split_cfg['mode']=='clip',
        augment=split_cfg['use_augmentation'],
    )
    label_transform = lambda images: torch.stack([torch.tensor(np.asarray(x)) for x in images], dim=0)

    import datasets
    if db_cfg['name'] == 'davis':
        dataset = datasets.DAVIS
    else:
        raise ValueError('Unknown dataset')

    clips_per_video = split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1
    db = dataset(
        subset=split_cfg['split'],
        full_res=db_cfg['full_res'],
        return_video=True,
        video_clip_duration=db_cfg['clip_duration'],
        joint_transform=joint_transform,
        video_transform=video_transform,
        return_labels=True,
        label_transform=label_transform,
        mode=split_cfg['mode'],
        clips_per_video=clips_per_video,
        shuffle=True,
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    drop_last = split_cfg['drop_last'] if 'drop_last' in split_cfg else True
    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size']  if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size']//split_cfg['clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None) and split_cfg['use_shuffle'],
        sampler=sampler,
        drop_last=drop_last
    )
    return loader


def build_dataloaders(cfg, num_workers, distributed, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = build_dataloader(cfg, cfg['train'], num_workers, distributed)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = build_dataloader(cfg, cfg['test'], num_workers, distributed)
    logger.add_line(str(test_loader.dataset))

    return train_loader, test_loader


def build_audiovisual_dataloader(db_cfg, split_cfg, num_workers, distributed):
    import numpy as np
    import torch.utils.data as data
    from datasets import preprocessing

    if 'horizon_only' not in db_cfg:
        db_cfg['horizon_only'] = False

    joint_transform = preprocessing.SpatialVideoCropTool(
        size=(db_cfg['crop_size'], db_cfg['crop_size']),
        hfov_lims=split_cfg['hfov_lims'],
        horizon_only=db_cfg['horizon_only'],
        margin=db_cfg['crop_margin'],
        pos=db_cfg['crop_method'],
        audio_input=db_cfg['audio_input'],
        num_crops=1 if db_cfg['use_temporal_augm'] else db_cfg['augm_per_clip'],
        random_flip=split_cfg['use_augmentation'],
    )
    video_transform = preprocessing.VideoPrep_CJ(
        augment=split_cfg['use_augmentation'],
        num_frames=int(db_cfg['video_fps'] * db_cfg['video_clip_duration']),
        pad_missing=True,
        random_color=True,
        random_flip=False,
    )

    audio_transforms = [
        preprocessing.AudioPrep(
            mono=db_cfg['audio_input']=='mono',
            duration=db_cfg['audio_clip_duration'],
            augment=split_cfg['use_augmentation']),
        preprocessing.LogMelSpectrogram(
            db_cfg['audio_fps'],
            n_mels=db_cfg['n_mels'],
            n_fft=db_cfg['n_fft'],
            hop_size=1. / db_cfg['spectrogram_fps'],
            normalize=True)
    ]

    import datasets
    if db_cfg['name'] == 'yt360':
        dataset = datasets.YT360Segm
    else:
        raise ValueError('Unknown dataset')

    clips_per_video = split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1
    db = dataset(
        subset=split_cfg['subset'],
        full_res=db_cfg['full_res'],
        return_video=True,
        video_clip_duration=db_cfg['video_clip_duration'],
        video_fps=db_cfg['video_fps'],
        return_audio=True,
        audio_clip_duration=db_cfg['audio_clip_duration'],
        audio_fps=db_cfg['audio_fps'],
        spect_fps=db_cfg['spectrogram_fps'],
        joint_transform=joint_transform,
        video_transform=video_transform,
        audio_transform=audio_transforms,
        mode=split_cfg['mode'],
        clips_per_video=clips_per_video,
        shuffle=True,
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    drop_last = split_cfg['drop_last'] if 'drop_last' in split_cfg else True
    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size'] if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size']//split_cfg['clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=drop_last
    )
    return loader


def build_audiovisual_dataloaders(cfg, num_workers, distributed, logger=None):
    train_loader = build_audiovisual_dataloader(cfg, cfg['train'], num_workers, distributed)
    test_loader = build_audiovisual_dataloader(cfg, cfg['test'], num_workers, distributed)
    return train_loader, test_loader


def log_dataset(train_loader, test_loader, logger=None):
    if logger is not None:
        logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
        logger.add_line(str(train_loader.dataset))
    if logger is not None:
        logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
        logger.add_line(str(test_loader.dataset))


def soft_nll_loss(inputs, targets):
    assert inputs.shape == targets.shape, 'size mismatch'
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    nll = (-targets * inputs.log()).sum(1)
    return nll.mean()


def iou(inputs, targets):
    assert inputs.shape == targets.shape
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    pred_mask = inputs > 0
    target_mask = targets > 0
    intersect = (pred_mask & target_mask).float().sum(1)
    union = (pred_mask | target_mask).float().sum(1)
    iou = intersect / union
    iou = iou[iou == iou]  # discard nan results
    return iou.mean()


def intersect_union(inputs, targets):
    n_class = inputs.size(1)
    # inputs = inputs.view(inputs.size(0), -1)
    # targets = targets.view(targets.size(0), -1)
    pred = inputs.argmax(1)
    intersect = []
    union = []
    for k in range(n_class):
        pred_k = pred == k
        target_k = targets == k
        intersect.append((pred_k & target_k).float().sum())
        union.append((pred_k | target_k).float().sum())
    return intersect, union


def accuracy(preds, labels):
    preds = preds.argmax(1)
    valid = (labels < 255)
    acc_sum = (valid * (preds == labels)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10) * 100.
    return acc, valid_sum


def class_accuracy(preds, labels):
    preds = preds.argmax(1)
    valid = (labels < 255)
    acc_sum = (valid * (preds == labels)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10) * 100.
    return acc, valid_sum


def mean_iou_with_unlabeled(preds, labels, numClass):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    preds = preds.argmax(1)
    preds[labels == 255] = 255

    # Compute area intersection:
    # miou = torch.zeros((numClass), dtype=torch.float32, device=preds.device)
    # for k in range(numClass):
    #     pred_k = preds == k
    #     target_k = labels == k
    #     intersect = (pred_k & target_k).float().flatten(1, 2).sum(1)
    #     union = (pred_k | target_k).float().flatten(1, 2).sum(1)
    #     iou = (intersect / union)
    #     iou = iou[iou == iou]
    #     miou[k] = iou.mean()
    # miou = miou[miou == miou]
    # return miou.mean()

    intersection = torch.zeros((preds.shape[0], numClass), dtype=torch.float32, device=preds.device)
    union = torch.zeros((preds.shape[0], numClass), dtype=torch.float32, device=preds.device)
    for k in range(numClass):
        pred_k = preds == k
        target_k = labels == k
        intersection[:, k] = (pred_k & target_k).float().flatten(1, 2).sum(1)
        union[:, k] = (pred_k | target_k).float().flatten(1, 2).sum(1)
    iou = intersection / union
    iou = iou[iou == iou]  # discard nan results
    return iou.mean() * 100.