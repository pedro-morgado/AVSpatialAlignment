import os
import json
import numpy as np
import torch.utils.data as data
from collections import defaultdict

from datasets.video_db import VideoDataset
from utils.ioutils import av_wrappers

ROOT = 'data/hmdb/'
ANNO_DIR = f'{ROOT}/ucfTrainTestlist/'
CACHE_DIR = 'datasets/cache/hmdb'


def get_subset(subset, split):
    import av
    cache_fn = f'{CACHE_DIR}/{subset}-{split}.json'
    if os.path.isfile(cache_fn):
        cache = json.load(open(cache_fn))
        return cache['files'], cache['labels'], cache['classes']

    # Classes
    classes = sorted([l.strip() for l in open(f'{ROOT}/classes.txt', 'r')])

    # Filter by subset
    files = set()
    nsplit = 0
    for cls in classes:
        for ln in open('{}/{}_test_split{}.txt'.format(ANNO_DIR, cls, split)):
            fn, ss = ln.strip().split()
            fn = '{}/{}'.format(cls, fn)
            if subset == 'train' and ss == '1' or subset == 'test' and ss == '2':
                nsplit += 1
                try:
                    av.open(f"{ROOT}/videos/{fn}")
                    files.add(fn)
                except UnicodeDecodeError:
                    pass

    files = sorted(list(files))
    labels = [classes.index(fn.split('/')[0]) for fn in files]

    cache = {'classes': classes, 'files': files, 'labels': labels}
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(cache, open(cache_fn, 'w'))
    return get_subset(subset, split)


class HMDB(VideoDataset):
    def __init__(self, subset,
                 full_res=True,
                 return_video=True,
                 video_fps=25.,
                 video_clip_duration=1.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=22050,
                 spect_fps=64,
                 audio_transform=None,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 shuffle=True,
                 ):

        self.name = 'HMDB-51'
        self.root = ROOT
        self.subset = subset

        subset, split = subset.split('-')
        split = int(split)

        files, labels, classes = get_subset(subset, split)

        video2clips = defaultdict(list)
        for i, fn in enumerate(files):
            video2clips[fn] = [i]

        self.classes = classes
        self.num_classes = len(self.classes)
        self.num_videos = len(files)

        super(HMDB, self).__init__(
            sampling='video',
            video2clips=video2clips,
            return_video=return_video,
            video_root='{}/videos'.format(ROOT),
            video_fns=files,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            return_audio=return_audio,
            audio_root='{}/videos'.format(ROOT),
            audio_fns=files,
            use_ambix=False,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            spect_fps=spect_fps,
            video_transform=video_transform,
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
    CROP_SIZE = 224

    video_transform = preprocessing.VideoPrep_MSC_CJ(crop=(CROP_SIZE, CROP_SIZE), augment=True, num_frames=16, pad_missing=True)
    audio_transform = preprocessing.AudioPrep(duration=CLIP_DURATION, augment=True)
    dataset = HMDB(
        subset='test-03',
        video_clip_duration=CLIP_DURATION,
        return_video=True,
        video_fps=VIDEO_FPS,
        video_transform=video_transform,
        return_audio=False,
        audio_fps=AUDIO_FPS,
        audio_transform=audio_transform,
        return_labels=True,
    )

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
        for bs in [32]:
            print('=' * 60)
            benchmark(batch_size=bs, num_workers=w)