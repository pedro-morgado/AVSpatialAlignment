import os
import json
import glob
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from datasets.video_db import VideoDataset

ROOT = 'data/yt360'
CACHE_DIR = 'datasets/cache/yt360'
ANNO_DIR = 'datasets/assets/yt360'


def get_metadata(subset='train'):
    cache_fn = f'{CACHE_DIR}/{subset}.json'
    if os.path.isfile(cache_fn):
        cache = json.load(open(cache_fn))
        return cache['video_fns'], cache['audio_fns'], cache['video2clips']

    video_fns = glob.glob(f'{ROOT}/video/*.mp4')
    video_files = {}
    for fn in video_fns:
        yid = fn.split('/')[-1][:11]
        t = int(fn.split('/')[-1].split('-')[-1].split('.')[0])
        video_files[(yid, t)] = fn.split('/')[-1]
    
    audio_fns = glob.glob(f'{ROOT}/audio/*.m4a')
    audio_files = {}
    for fn in audio_fns:
        yid = fn.split('/')[-1][:11]
        t = int(fn.split('/')[-1].split('-')[-1].split('.')[0])
        audio_files[(yid, t)] = fn.split('/')[-1]

    files = {}
    for k in video_files:
        if k in audio_files:
            files[k] = [video_files[k], audio_files[k]]

    def create_cache(files_to_cache, output_fn):
        video_fns, audio_fns, video2clips = [], [], defaultdict(list)
        for i, k in enumerate(files_to_cache):
            video_fns += [files_to_cache[k][0]]
            audio_fns += [files_to_cache[k][1]]
            video2clips[k[0]] += [i]
        cache = {'video_fns': video_fns, 'audio_fns': audio_fns, 'video2clips': video2clips}
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        json.dump(cache, open(output_fn, 'w'))
    
    anno_fn = f'{ANNO_DIR}/{subset}.txt'
    with open(anno_fn, 'r') as f:
        yid = [line.rstrip() for line in f]
    
    subset_files = {k: files[k] for k in files if k[0] in yid}
    create_cache(subset_files, cache_fn)

    return get_metadata(subset)


class YT360(VideoDataset):
    def __init__(self,
                 subset='train',
                 full_res=False,
                 sampling='video',
                 return_video=True,
                 video_clip_duration=0.5,
                 video_fps=16.,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=24000,
                 spect_fps=100,
                 joint_transform=None,
                 video_transform=None,
                 audio_transform=None,
                 return_position=False,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 augm_per_clip=1,
                 use_temporal_augm=True,
                 use_spatial_augm=False,
                 misalign=0,
                 rotate_mode='quat',
                 shuffle=True,
                 ):

        self.name = 'YT360'
        self.root = ROOT
        assert full_res == False
        assert return_labels == False

        video_fns, audio_fns, video2clips = get_metadata(subset=subset)
        self.num_clips = len(video2clips)
        self.num_videos = len(video_fns)

        super(YT360, self).__init__(
            sampling=sampling,
            video2clips=video2clips,
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root=f'{ROOT}/video',
            video_fns=video_fns,
            video_fps=video_fps,
            return_audio=return_audio,
            audio_clip_duration=audio_clip_duration,
            audio_root=f'{ROOT}/audio',
            audio_fns=audio_fns,
            use_ambix=True,
            audio_fps=audio_fps,
            spect_fps=spect_fps,
            return_position=return_position,
            joint_transform=joint_transform,
            video_transform=video_transform,
            audio_transform=audio_transform,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
            augm_per_clip=augm_per_clip,
            use_temporal_augm=use_temporal_augm,
            use_spatial_augm=use_spatial_augm,
            misalign=misalign,
            rotate_mode=rotate_mode,
            shuffle=shuffle
        )

class SegmTransform:
    def __init__(self):
        filtered_cls_idx = np.loadtxt('datasets/assets/segm-meta/segmentation-filtered-class-idx.txt').astype(int).tolist()
        segm_classes = [l.strip() for l in open('datasets/assets/segm-meta/segmentation-classes.txt')]
        self.label_map = np.array([filtered_cls_idx.index(i) if i in filtered_cls_idx else 255 for i in range(256)])
        self.segm_classes = [segm_classes[i] for i in filtered_cls_idx]

    def __call__(self, x):
        return self.label_map[x]

class YT360Segm(VideoDataset):
    def __init__(self,
                 subset='train',
                 full_res=False,
                 sampling='video',
                 return_video=True,
                 video_clip_duration=0.5,
                 video_fps=16.,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=24000,
                 spect_fps=100,
                 joint_transform=None,
                 video_transform=None,
                 audio_transform=None,
                 return_position=False,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 augm_per_clip=1,
                 use_temporal_augm=True,
                 use_spatial_augm=False,
                 shuffle=True,
                 ):

        self.name = 'YT360'
        self.root = ROOT
        assert full_res == False
        assert return_labels == False

        video_fns, audio_fns, video2clips = get_metadata(subset=subset)
        video2clips, segmentation_dirs = defaultdict(list), [None]*len(video_fns)
        for i, fn in enumerate(video_fns):
            segmentation_dirs[i] = '.'.join(fn.split('.')[:-1])
            video2clips[fn[:11]] += [i]

        segmentation_transform = SegmTransform()

        self.num_clips = len(video2clips)
        self.num_videos = len(video_fns)

        super(YT360Segm, self).__init__(
            sampling=sampling,
            video2clips=video2clips,
            return_video=return_video,
            video_clip_duration=video_clip_duration,
            video_root=f'{ROOT}/video',
            video_fns=video_fns,
            video_fps=video_fps,
            return_audio=return_audio,
            audio_clip_duration=audio_clip_duration,
            audio_root=f'{ROOT}/audio',
            audio_fns=audio_fns,
            use_ambix=True,
            audio_fps=audio_fps,
            spect_fps=spect_fps,
            return_segmentation=True,
            segmentation_root=f"{ROOT}/segmentation",
            segmentation_dirs=segmentation_dirs,
            segmentation_transform=segmentation_transform,
            return_position=return_position,
            joint_transform=joint_transform,
            video_transform=video_transform,
            audio_transform=audio_transform,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
            augm_per_clip=augm_per_clip,
            use_temporal_augm=use_temporal_augm,
            use_spatial_augm=use_spatial_augm,
            shuffle=shuffle
        )


def benchmark(batch_size, num_workers):
    import time
    from datasets import preprocessing

    VIDEO_CLIP_DURATION = 0.5
    VIDEO_FPS = 16.
    VIDEO_FRAMES = int(VIDEO_CLIP_DURATION * VIDEO_FPS)
    AUDIO_CLIP_DURATION = 2.0
    AUDIO_FPS = 24000
    SPECTROGRAM_FPS = 64
    CROP_SIZE = 112

    joint_transform = preprocessing.SpatialVideoCropTool(size=(CROP_SIZE, CROP_SIZE))
    video_transform = preprocessing.VideoPrep_CJ(augment=True, pad_missing=True, num_frames=VIDEO_FRAMES)
    audio_transform = [
        preprocessing.AudioPrep(duration=AUDIO_CLIP_DURATION, augment=True, mono=True),
        preprocessing.LogMelSpectrogram(AUDIO_FPS, n_fft=2048, n_mels=128, hop_size=1. / SPECTROGRAM_FPS, normalize=True)
    ]
    dataset = YT360(
        subset='train',
        joint_transform=joint_transform,
        return_video=True,
        video_clip_duration=VIDEO_CLIP_DURATION,
        video_fps=VIDEO_FPS,
        video_transform=video_transform,
        return_audio=True,
        audio_clip_duration=AUDIO_CLIP_DURATION,
        audio_fps=AUDIO_FPS,
        spect_fps=SPECTROGRAM_FPS,
        audio_transform=audio_transform,
        clips_per_video=1000,
        mode='clip'
    )
    print(dataset)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    tt = time.time()
    read_times = []
    frames_per_clip = int(VIDEO_FPS * VIDEO_CLIP_DURATION)
    for idx, batch in enumerate(loader):
        print('timing')
        for k in batch:
            if 'time' in k:
                print(f'{k:20}: {batch[k].mean()}')

        if idx < 25:
            tt = time.time()
            continue
        read_times.append(time.time() - tt)
        tt = time.time()

        secs_per_clip = np.mean(read_times) / batch_size
        print('Iter {:03d} of {:04d} | Secs per batch {:.3f} | Clips per sec {:.3f} | Frames per sec  {:.3f}'.format(
            idx, len(loader), secs_per_clip * batch_size, 1. / secs_per_clip, frames_per_clip / secs_per_clip
        ))

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