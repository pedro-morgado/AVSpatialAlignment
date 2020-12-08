import argparse
import time
import yaml
import torch

from utils import main_utils, segm_eval_utils
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='Evaluation on Video Segmentation')
parser.add_argument('cfg', metavar='CFG', help='eval config file')
parser.add_argument('model_cfg', metavar='MODEL_CFG', help='model config file')
parser.add_argument('--checkpoint-dir', metavar='CKP', help='checkpoint')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--num-workers', default=None, type=int)
parser.add_argument('--port', default='1234')


def scheduler():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.num_workers is not None:
        cfg['num_workers'] = args.num_workers
    if args.debug:
        cfg['dataset']['batch_size'] = 4
        cfg['num_workers'] = 0

    ngpus = torch.cuda.device_count()
    main(0, ngpus, args, cfg)


def main(gpu, ngpus, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare for training
    model_cfg, criterion_cfg, eval_dir, logger = segm_eval_utils.prepare_environment(args, cfg)
    model, head_params, ckp_manager = segm_eval_utils.build_model(model_cfg, criterion_cfg, cfg['model'], eval_dir, args, logger)
    model = segm_eval_utils.distribute_model_to_cuda(model, args, cfg)
    if cfg['optimizer']['head_only']:
        try:
            model.video_model
        except AttributeError:
            model.head_parameters = model.module.head_parameters
            model.freeze_backbone = model.module.freeze_backbone
        model.freeze_backbone()
        optimizer, scheduler = main_utils.build_optimizer(head_params, cfg['optimizer'], logger)
    else:
        optimizer, scheduler = main_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)
    train_loader, test_loader = segm_eval_utils.build_audiovisual_dataloaders(cfg['dataset'], cfg['num_workers'], False, logger)

    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume'] or cfg['test_only']:
        start_epoch = ckp_manager.restore(model, optimizer, scheduler, restore_last=True, logger=logger)

    ######################### LOGGING #########################
    segm_eval_utils.log_model(model, logger)
    segm_eval_utils.log_dataset(train_loader, test_loader, logger)

    ######################### TRAINING #########################
    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        for epoch in range(start_epoch, end_epoch):
            train_loader.dataset.shuffle_dataset()
            test_loader.dataset.shuffle_dataset()

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(scheduler.get_lr()))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            iou, acc = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, scheduler, epoch, eval_metric=iou)
            scheduler.step()

    ######################### TESTING #########################
    logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
    cfg['dataset']['test']['clips_per_video'] = 10
    _, test_loader = segm_eval_utils.build_audiovisual_dataloaders(cfg['dataset'], cfg['num_workers'], False, logger)
    iou, acc = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)

    ######################### LOG RESULTS #########################
    logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
    logger.add_line('IoU: {:6.4f}'.format(iou*100))
    logger.add_line('Pix Acc: {:6.4f}'.format(iou*100))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    batch_time = main_utils.AverageMeter('Time', ':6.3f', window_size=100)
    data_time = main_utils.AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = main_utils.AverageMeter('Loss', ':.4e')
    acc_meter = main_utils.AverageMeter('Acc', ':6.2f')
    iou_meter = main_utils.AverageMeter('mIoU', ':6.2f')
    progress = main_utils.ProgressMeter(
        len(loader), meters=[batch_time, data_time, loss_meter, acc_meter, iou_meter],
        phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    model.train(phase == 'train')

    if phase in {'test_dense', 'test'}:
        model = segm_eval_utils.BatchWrapper(model, cfg['dataset']['batch_size'])

    end = time.time()
    class_acc = []
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        if args.debug and it == 0:
            plot_batch(sample)

        video = sample['video']
        audio = sample['audio']
        target = sample['segmentation'].long()

        # Skip iteration if there are no valid labels in the batch
        valid = (target != 255).flatten(1, -1).sum(-1) > 0
        if valid.sum() == 0:
            continue

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
            audio = audio.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute outputs
        if phase == 'train':
            output = model(video, audio)
        else:
            with torch.no_grad():
                output = model(video, audio)

        # downsampled target map for training
        output = output.flatten(0, 1)
        target = target.flatten(0, 1)
        outputs_up = torch.nn.functional.interpolate(output, target.shape[1:], mode='bilinear', align_corners=False)

        # compute loss and measure accuracy
        loss = torch.nn.functional.cross_entropy(outputs_up, target, reduction="mean", ignore_index=255)

        import numpy as np
        with torch.no_grad():
            loss_meter.update(loss.item(), target.size(0))
            acc, _ = segm_eval_utils.accuracy(outputs_up, target)

            acc_meter.update(acc.mean().item(), target.size(0))
            iou = segm_eval_utils.mean_iou_with_unlabeled(outputs_up, target, cfg['model']['args']['num_classes'])
            iou_meter.update(iou.mean().item(), target.size(0))

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

    return iou_meter.avg, acc_meter.avg


def plot_batch(sample):
    import matplotlib.pyplot as plt
    import numpy as np

    audio = sample['audio']
    video = sample['video']
    segmentation = sample['segmentation']

    bs, nv, ncv, ntv, w, h = video.shape
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    f, ax = plt.subplots(bs, 5)
    for n in range(bs):
        vid = video[n, 0].permute(1, 2, 3, 0).cpu().numpy()
        snd = audio[n, 0].cpu().numpy()
        seg = segmentation[n, 0].cpu().numpy()
        ax[n, 0].imshow((np.clip(vid[0] * std + mean, 0, 1) * 255).astype(np.uint8))
        ax[n, 0].set_axis_off()
        ax[n, 1].imshow((np.clip(vid[ntv//2] * std + mean, 0, 1) * 255).astype(np.uint8))
        ax[n, 1].set_axis_off()
        ax[n, 2].imshow((np.clip(vid[-1] * std + mean, 0, 1) * 255).astype(np.uint8))
        ax[n, 2].set_axis_off()
        ax[n, 3].imshow(seg)
        ax[n, 3].set_axis_off()
        ax[n, 4].imshow(snd[0])
        ax[n, 4].set_axis_off()
    plt.show()


if __name__ == '__main__':
    scheduler()
