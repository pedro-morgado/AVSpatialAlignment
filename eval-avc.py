import argparse
import time
import yaml
import torch

from utils import main_utils, eval_utils
import torch.multiprocessing as mp


parser = argparse.ArgumentParser(description='Evaluation on Audio-Visual Correspondance')
parser.add_argument('cfg', metavar='CFG', help='eval config file')
parser.add_argument('model_cfg', metavar='MODEL_CFG', help='model config file')
parser.add_argument('--checkpoint-dir', metavar='CKP', help='checkpoint')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--crop-acc', action='store_true')


def scheduler():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.debug:
        cfg['dataset']['batch_size'] = 4
        cfg['num_workers'] = 0

    ngpus = torch.cuda.device_count()
    if args.distributed:
        mp.spawn(main, nprocs=ngpus, args=(ngpus, args, cfg))
    else:
        main(0, ngpus, args, cfg)


def main(gpu, ngpus, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare for training
    model_cfg, eval_dir, logger = eval_utils.prepare_environment(args, cfg)
    if 'scratch' not in cfg:
        cfg['scratch'] = False
    if 'ft_all' not in cfg:
        cfg['ft_all'] = False
    model, ckp_manager, ckp = eval_utils.build_model(model_cfg, cfg['model'], eval_dir, args, logger, return_ckp=True, scratch=cfg['scratch'])
    params = list(model.parameters()) if cfg['ft_all'] else model.head_params()

    if cfg['use_transf'] != 'none':
        loss_cfg = yaml.safe_load(open(args.model_cfg))['loss']
        align_criterion = main_utils.build_criterion(loss_cfg, logger=logger).cuda(gpu)
        align_criterion.load_state_dict(ckp['train_criterion'])
        if type(align_criterion).__name__ == 'MultiTask':
            align_criterion = align_criterion.losses[0]  # MultiTask
        if cfg['ft_all']:
            params += list(align_criterion.parameters())
    else:
        align_criterion = None

    optimizer, scheduler = main_utils.build_optimizer(params, cfg['optimizer'], logger)
    train_loader, test_loader = build_dataloaders(cfg['dataset'], cfg['num_workers'], args.distributed, logger)

    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if 'resume' in cfg:
        args.resume = cfg['resume']
    if 'test_only' in cfg:
        args.test_only = cfg['test_only']
    if args.resume or args.test_only:
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, align_criterion, restore_last=True, logger=logger)

    ######################### TRAINING #########################
    if not args.test_only:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        for epoch in range(start_epoch, end_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)
            train_loader.dataset.shuffle_dataset()
            test_loader.dataset.shuffle_dataset()

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(scheduler.get_lr()))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger, align_criterion)
            top1 = run_phase('test', test_loader, model, None, epoch, args, cfg, logger, align_criterion)
            ckp_manager.save(model, optimizer, scheduler, epoch, criterion=align_criterion, eval_metric=top1)
            scheduler.step()

    ######################### TESTING #########################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    top1 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger, align_criterion)

    ######################### LOG RESULTS #########################
    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    logger.add_line('Clip@1: {:6.2f}'.format(top1))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger, align_criterion):
    batch_time = main_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = main_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = main_utils.AverageMeter('Loss', ':.4e')
    acc_meter = main_utils.AverageMeter('Acc@1', ':6.2f')
    progress = main_utils.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, acc_meter],
                                        phase=phase, epoch=epoch, logger=logger)

    try:
        audio_channels = model.audio_feat.conv1[0].in_channels
        model.extract_features
    except AttributeError:
        audio_channels = model.module.audio_feat.conv1[0].in_channels
        model.extract_features = model.module.extract_features
        model.classify = model.module.classify

    # switch to train/test mode
    model.train(phase == 'train' and cfg['ft_all'])
    model.classifier.train(phase == 'train')
    if align_criterion is not None:
        align_criterion.train(phase == 'train' and cfg['ft_all'])

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        bs, n_aug = sample['video'].shape[:2]

        video = sample['video'].flatten(0, 1)
        audio = sample['audio'].flatten(0, 1)

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
            audio = audio.cuda(args.gpu, non_blocking=True)

        # compute outputs
        with torch.set_grad_enabled(phase == 'train' and cfg['ft_all']):
            video_emb, audio_emb = model.extract_features(video, audio)
            video_emb, audio_emb = video_emb.view(bs, n_aug, -1), audio_emb.view(bs, n_aug, -1)
            if align_criterion is not None:
                audio_emb_pred, video_emb_pred = align_criterion.predict(video_emb, audio_emb)
                if cfg['use_transf'] == 'concat':
                    video_emb = torch.cat((video_emb, audio_emb_pred), -1)
                    audio_emb = torch.cat((audio_emb, video_emb_pred), -1)
                elif cfg['use_transf'] == 'parallel':
                    video_emb = torch.cat((video_emb, audio_emb_pred), 1)
                    audio_emb = torch.cat((audio_emb, video_emb_pred), 1)
                    n_aug *= 2
                elif cfg['use_transf'] == 'audio':
                    video_emb = audio_emb_pred
                elif cfg['use_transf'] == 'video':
                    audio_emb = video_emb_pred
                else:
                    raise Exception('unsupported feat type: %s' % cfg['use_transf'])

            video_emb, audio_emb, labels = create_avc_pairs(video_emb, audio_emb, args)
            video_emb = video_emb.flatten(0, 1)
            audio_emb = audio_emb.flatten(0, 1)

        with torch.set_grad_enabled(phase == 'train'):
            if 'audio_only' in cfg and cfg['audio_only']:
                logits = model.classify(audio_emb, audio_emb)
            else:
                logits = model.classify(video_emb, audio_emb)

            # compute loss and measure accuracy
            crop_labels = labels.unsqueeze(1).repeat(1, n_aug).view(-1)
            loss = criterion(logits, crop_labels)

        with torch.no_grad():
            if args.crop_acc:
                acc = main_utils.accuracy(logits, crop_labels, topk=(1, ))[0]
            else:
                pred_video = logits.view(2 * bs, n_aug, -1).mean(1)
                acc = main_utils.accuracy(pred_video, labels, topk=(1, ))[0]
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % cfg['print_freq'] == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)

    if args.distributed:
        main_utils.synchronize_meters(progress, args.gpu)
        progress.display(len(loader) * args.world_size)

    return acc_meter.avg


def create_avc_pairs(video_embs, audio_embs, args):
    bs = video_embs.shape[0]
    rnd_idx = torch.randint(0, bs - 1, (bs,))
    rnd_idx = rnd_idx + (rnd_idx >= torch.arange(0, bs)).int()
    video_embs = torch.cat((video_embs, video_embs), 0)
    audio_embs = torch.cat((audio_embs, audio_embs[rnd_idx]), 0)
    labels = torch.cat((torch.zeros(bs, ), torch.ones(bs, )), 0).long()
    if args.gpu is not None:
        labels = labels.cuda(args.gpu, non_blocking=True)
    return video_embs, audio_embs, labels


def build_dataloaders(cfg, num_workers, distributed, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = main_utils.build_dataloader(cfg, cfg['train'], num_workers, distributed)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = main_utils.build_dataloader(cfg, cfg['test'], num_workers, distributed)
    logger.add_line(str(test_loader.dataset))

    return train_loader, test_loader


if __name__ == '__main__':
    scheduler()