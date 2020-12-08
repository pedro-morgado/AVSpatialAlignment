import os
import json
import numpy as np
import torch.utils.data as data
from collections import defaultdict

from datasets.video_db import VideoDataset

ROOT = 'data/ucf101/'
ANNO_DIR = f'{ROOT}/ucfTrainTestlist/'
CACHE_DIR = 'datasets/cache/ucf101'


def get_metadata():
    classes_fn = f'{ANNO_DIR}/classInd.txt'
    classes = [l.strip().split()[1] for l in open(classes_fn)]

    cache_fn = CACHE_DIR+'/meta.json'
    if os.path.isfile(cache_fn):
        ucf_meta = json.load(open(cache_fn))
        return ucf_meta, classes

    all_files, all_labels = [], []
    for list_fn in ['trainlist01.txt', 'testlist01.txt']:
        for ln in open('{}/{}'.format(ANNO_DIR, list_fn)):
            fn = ln.strip().split()[0]
            lbl = classes.index(fn.split('/')[0])
            all_files.append(fn)
            all_labels.append(lbl)

    ucf_meta = []
    for path, lbl in zip(all_files, all_labels):
        ucf_meta.append({
            'fn': path,
            'label': lbl,
            'label_str': classes[lbl],
        })

    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(ucf_meta, open(cache_fn, 'w'))
    return ucf_meta, classes


def filter_samples(meta, subset=None, ignore_video_only=False):
    # Filter by subset
    if subset is not None:
        subset = set([ln.strip().split()[0] for ln in open('{}/{}.txt'.format(ANNO_DIR, subset))])
        meta = [m for m in meta if '{}/{}'.format(m['label_str'], m['fn'].split('/')[-1]) in subset]

    # Filter videos with no audio
    if ignore_video_only:
        meta = [m for m in meta if m['audio_start'] != -1 or m['audio_duration'] != -1]

    return meta


class UCF(VideoDataset):
    def __init__(self, subset,
                 full_res=True,
                 return_video=True,
                 video_clip_duration=0.5,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=0.5,
                 audio_fps=22050,
                 spect_fps=64,
                 audio_transform=None,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 shuffle=False
                 ):

        self.name = 'UCF-101'
        self.root = ROOT
        self.subset = subset

        meta, classes = get_metadata()
        meta = filter_samples(meta, subset=subset, ignore_video_only=return_audio)
        fns = [m['fn'] for m in meta]
        labels = [m['label'] for m in meta]
        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(meta)

        video2clips = defaultdict(list)
        for i, fn in enumerate(fns):
            vid = fn.split('/')[-1].split('.')[0]
            video2clips[vid] = [i]

        super(UCF, self).__init__(
            sampling='video',
            video2clips=video2clips,
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root='{}/data'.format(ROOT),
            video_fns=fns,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root='{}/data'.format(ROOT),
            audio_fns=fns,
            use_ambix=False,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            spect_fps=spect_fps,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
            shuffle=shuffle,
        )


def benchmark(batch_size, num_workers):
    import time
    from datasets import preprocessing

    CLIP_DURATION = 1.
    VIDEO_FPS = 16.
    AUDIO_FPS = 48000
    SPECTROGRAM_FPS = 64.
    FRAME_SIZE = 256
    CROP_SIZE = 224

    video_transform = preprocessing.VideoPreprocessing(resize=FRAME_SIZE, crop=(CROP_SIZE, CROP_SIZE), augment=True)
    audio_transform = preprocessing.AudioPrep(duration=CLIP_DURATION, missing_as_zero=True, augment=True)
    dataset = UCF('trainlist01',
                clip_duration=CLIP_DURATION,
                return_video=True,
                video_fps=VIDEO_FPS,
                video_transform=video_transform,
                return_audio=False,
                audio_fps=AUDIO_FPS,
                audio_transform=audio_transform,
                return_labels=True,
                missing_audio_as_zero=True)

    print(dataset)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    tt = time.time()
    read_times = []
    frames_per_clip = int(VIDEO_FPS * CLIP_DURATION)
    for idx, batch in enumerate(loader):
        read_times.append(time.time() - tt)
        tt = time.time()

        secs_per_clip = np.mean(read_times[1:]) / batch_size
        print('Iter {:03d} | Secs per batch {:.3f} | Clips per sec {:.3f} | Frames per sec  {:.3f}'.format(
            idx, secs_per_clip * batch_size, 1. / secs_per_clip, frames_per_clip / secs_per_clip
        ))
        if idx > 100:
            break

    secs_per_clip = np.mean(read_times[1:]) / batch_size

    print('')
    print('Num workers     | {}'.format(num_workers))
    print('Batch size      | {}'.format(batch_size))
    print('Frames per clip | {}'.format(frames_per_clip))
    print('Secs per batch  | {:.3f}'.format(secs_per_clip * batch_size))
    print('Clips per sec   | {:.3f}'.format(1. / secs_per_clip))
    print('Frames per sec  | {:.3f}'.format(frames_per_clip / secs_per_clip))


if __name__ == '__main__':
    for w in [1]:
        for bs in [62]:
            print('=' * 60)
            benchmark(batch_size=bs, num_workers=w)