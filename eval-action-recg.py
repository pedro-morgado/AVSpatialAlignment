import argparse
import time
import yaml
import torch

from utils import main_utils, eval_utils


parser = argparse.ArgumentParser(description='Evaluation on Action Recognition')
parser.add_argument('cfg', metavar='CFG', help='eval config file')
parser.add_argument('model_cfg', metavar='MODEL_CFG', help='model config file')
parser.add_argument('--checkpoint-dir', metavar='CKP', help='checkpoint')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--port', default='1234')


def scheduler():
    import os
    os.system("nvidia-smi")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['dataset']['batch_size'] = 4
        cfg['num_workers'] = 0

    ngpus = torch.cuda.device_count()
    main(0, ngpus, args, cfg)


def main(gpu, ngpus, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare for training
    model_cfg, eval_dir, logger = eval_utils.prepare_environment(args, cfg)
    model, ckp_manager = eval_utils.build_model(model_cfg, cfg['model'], eval_dir, args, logger)
    optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], False, logger)

    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume'] or cfg['test_only']:
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True, logger=logger)

    ######################### TRAINING #########################
    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        for epoch in range(start_epoch, end_epoch):
            train_loader.dataset.shuffle_dataset()
            test_loader.dataset.shuffle_dataset()

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(scheduler.get_lr()))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, scheduler, epoch, eval_metric=top1)
            scheduler.step()

    ######################### TESTING #########################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 25
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], False, logger)
    top1, top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)
    top1_dense, top5_dense = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)

    ######################### LOG RESULTS #########################
    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    logger.add_line('Clip@1: {:6.2f}'.format(top1))
    logger.add_line('Clip@5: {:6.2f}'.format(top5))
    logger.add_line('Video@1: {:6.2f}'.format(top1_dense))
    logger.add_line('Video@5: {:6.2f}'.format(top5_dense))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    batch_time = main_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = main_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = main_utils.AverageMeter('Loss', ':.4e')
    top1_meter = main_utils.AverageMeter('Acc@1', ':6.2f')
    top5_meter = main_utils.AverageMeter('Acc@5', ':6.2f')
    progress = main_utils.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, top1_meter, top5_meter],
                                        phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    model.train(phase == 'train')
    if phase == 'train' and 'freeze_bn' in cfg and cfg['freeze_bn']:
        for m in model.modules():
            if 'BatchNorm' in m.__class__.__name__:
                m.eval()

    if phase in {'test_dense', 'test'}:
        model = eval_utils.BatchWrapper(model, cfg['dataset']['batch_size'])

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video = sample['video'][:, 0]
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = torch.flatten(video, 0, 1)

        # compute outputs
        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss and measure accuracy
        if phase == 'test_dense':
            labels_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, labels_tiled)
        else:
            loss = criterion(logits, target)

        with torch.no_grad():
            confidence = softmax(logits)
            if phase == 'test_dense':
                confidence = confidence.view(batch_size, clips_per_sample, -1).mean(1)

            acc1, acc5 = main_utils.accuracy(confidence, target, topk=(1, 5))
            loss_meter.update(loss.item(), target.size(0))
            top1_meter.update(acc1[0], target.size(0))
            top5_meter.update(acc5[0], target.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % 100 == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)

    return top1_meter.avg, top5_meter.avg


if __name__ == '__main__':
    scheduler()
