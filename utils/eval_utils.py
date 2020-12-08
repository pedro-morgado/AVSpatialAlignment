import os
import yaml
from utils import main_utils
import torch
import torch.distributed as dist
import torch.nn as nn


def prepare_environment(args, cfg):
    model_cfg = yaml.safe_load(open(args.model_cfg))['model']
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

    return model_cfg, eval_dir, logger


def build_model(feat_cfg, cfg, eval_dir, args, logger=None, return_ckp=False, scratch=False):
    import models
    pretrained_net = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])

    if scratch:
        ckp = None
    else:
        checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
        ckp = torch.load(checkpoint_fn, map_location='cpu')
        model_ckp = ckp['state_dict'] if 'state_dict' in ckp else ckp['model']
        try:
            pretrained_net.load_state_dict({k.replace('module.', ''): model_ckp[k] for k in model_ckp})
        except RuntimeError:  # load video model only
            model_dict = pretrained_net.video_model.state_dict()
            model_ckp = {k.replace('module.model.', ''): v for k, v in model_ckp.items()}
            model_dict.update({k: v for k, v in model_ckp.items() if k in model_dict})
            pretrained_net.video_model.load_state_dict(model_dict)

    if cfg['name'] == 'lreg_classifier':
        pretrained_net = pretrained_net.video_model
        model = ClassificationWrapper(feature_extractor=pretrained_net, **cfg['args'])
    elif cfg['name'] == 'avc_wrapper':
        model = AVCWrapper(video_feat=pretrained_net.video_model, audio_feat=pretrained_net.audio_model, **cfg['args'])
    else:
        raise ValueError
    ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)

    if logger is not None:
        logger.add_line("=" * 30 + "   Model   " + "=" * 30)
        logger.add_line(str(model))
        logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
        logger.add_line(main_utils.parameter_description(model))
        logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
        logger.add_line("File: {}\nEpoch: {}".format(checkpoint_fn, ckp['epoch']))

    model = distribute_model_to_cuda(model, args, cfg)
    if return_ckp:
        return model, ckp_manager, ckp

    return model, ckp_manager


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, scheduler, epoch, criterion=None, eval_metric=0.):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        ckp = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if criterion is not None:
            ckp['criterion'] = criterion.state_dict()
        main_utils.save_checkpoint(state=ckp, is_best=is_best, model_dir=self.checkpoint_dir)

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

    def restore(self, model, optimizer=None, scheduler=None, criterion=None, restore_last=False, restore_best=False, logger=None):
        start_epoch = 0
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        if os.path.exists(checkpoint_fn):
            ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
            start_epoch = ckp['epoch']
            model.load_state_dict(ckp['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(ckp['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(ckp['scheduler'])
            if criterion is not None:
                try:
                    criterion.load_state_dict(ckp['criterion'].state_dict())
                except AttributeError:
                    criterion.load_state_dict(ckp['criterion'])
            if logger is not None:
                logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(checkpoint_fn, start_epoch))
        else:
            if logger is not None:
                logger.add_line("No checkpoint found at '{}'.".format(checkpoint_fn))

        return start_epoch


class ClassificationWrapper(torch.nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_name, feat_dim, pooling_op=None, use_dropout=False, dropout=0.5):
        super(ClassificationWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.feat_name = feat_name
        self.use_dropout = use_dropout
        if pooling_op is not None:
            self.pooling = eval('torch.nn.'+pooling_op)
        else:
            self.pooling = None
        if use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(feat_dim, n_classes)

    def forward(self, *inputs):
        emb = self.feature_extractor(*inputs, return_embs=True)[self.feat_name]
        emb_pool = self.pooling(emb) if self.pooling is not None else emb
        emb_pool = emb_pool.view(inputs[0].shape[0], -1)
        if self.use_dropout:
            emb_pool = self.dropout(emb_pool)
        logit = self.classifier(emb_pool)
        return logit


class AVCWrapper(torch.nn.Module):
    def __init__(self, video_feat, audio_feat, use_dropout=False, dropout=0.5):
        super(AVCWrapper, self).__init__()
        self.video_feat = video_feat
        self.audio_feat = audio_feat
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.video_feat.out_dim + self.audio_feat.out_dim, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 2),
        )

    def head_params(self):
        return list(self.classifier.parameters())

    def extract_features(self, video, audio):
        video_emb = self.video_feat(video).flatten(1, -1)
        audio_emb = self.audio_feat(audio).flatten(1, -1)
        return video_emb, audio_emb

    def classify(self, video_emb, audio_emb):
        emb = torch.cat((video_emb, audio_emb), 1)
        if self.use_dropout:
            emb = self.dropout(emb)
        logit = self.classifier(emb)
        return logit

    def forward(self, video, audio):
        video_emb = self.video_feat(video).flatten(1, -1)
        audio_emb = self.audio_feat(audio).flatten(1, -1)
        emb = torch.cat((video_emb, audio_emb), 1)
        if self.use_dropout:
            emb = self.dropout(emb)
        logit = self.classifier(emb)
        return logit


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i+self.batch_size])]
        return torch.cat(outs, 0)


def distribute_model_to_cuda(model, args, cfg):
    torch.cuda.set_device(args.gpu)
    model = model.cuda()

    return model


def build_dataloader(db_cfg, split_cfg, num_workers, distributed):
    import torch.utils.data as data
    from datasets import preprocessing

    video_transform = preprocessing.VideoPrep_MSC_CJ(
        crop=(db_cfg['crop_size'], db_cfg['crop_size']),
        num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
        pad_missing=split_cfg['mode']=='clip',
        augment=split_cfg['use_augmentation'],
        color=(0.4, 0.4, 0.4, 0.2) if 'color' not in db_cfg else db_cfg['color'],
    )

    import datasets
    if db_cfg['name'] == 'ucf101':
        dataset = datasets.UCF
    elif db_cfg['name'] == 'hmdb51':
        dataset = datasets.HMDB
    elif db_cfg['name'] == 'kinetics':
        dataset = datasets.Kinetics
    else:
        raise ValueError('Unknown dataset')

    clips_per_video = split_cfg['clips_per_video'] if 'clips_per_video' in split_cfg else 1
    db = dataset(
        subset=split_cfg['split'].format(fold=db_cfg['fold']),
        return_video=True,
        video_clip_duration=db_cfg['clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
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

    logger.add_line("=" * 30 + "   Dense DB   " + "=" * 30)
    dense_loader = build_dataloader(cfg, cfg['test_dense'], num_workers, distributed)
    logger.add_line(str(dense_loader.dataset))

    return train_loader, test_loader, dense_loader
