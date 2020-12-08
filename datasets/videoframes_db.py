import random
import torch
import numpy as np
import torch.utils.data as data
from collections import defaultdict
from PIL import Image


def chararray(fn_list):
    charr = np.chararray(len(fn_list), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]
    return charr


class VideoFrameDataset(data.Dataset):
    def __init__(self,
                 sampling='video',
                 video2clips=None,
                 return_video=True,
                 video_root=None,
                 video_fns=None,
                 video_lengths=None,
                 video_clip_duration=1,
                 joint_transform=None,
                 video_transform=None,
                 return_labels=False,
                 labels=None,
                 label_root=None,
                 label_fns=None,
                 label_transform=None,
                 mode='clip',
                 augm_per_clip=1,
                 clips_per_video=1,
                 max_offsync_augm=0,
                 shuffle=True,
                 ):
        super(VideoFrameDataset, self).__init__()

        assert sampling in {'video', 'clip'}
        self.sampling = sampling
        self.video2clips = video2clips
        self.video_ids = list(video2clips.keys())
        self.num_videos = len(video2clips)
        self.num_clips = sum([len(video2clips[v]) for v in video2clips])
        self.num_samples = self.num_videos if sampling == 'video' else self.num_clips

        # Video config
        self.return_video = return_video
        self.video_root = video_root
        assert video_clip_duration > 0
        self.video_clip_duration = video_clip_duration
        if return_video:
            self.video_fns = chararray(video_fns)
            self.video_lengths = video_lengths

        # Transforms
        if joint_transform is None:
            joint_transform = []
        if not isinstance(joint_transform, list):
            joint_transform = [joint_transform]
        self.joint_transform = joint_transform

        if video_transform is None:
            video_transform = []
        if not isinstance(video_transform, list):
            video_transform = [video_transform]
        self.video_transform = video_transform

        if label_transform is None:
            label_transform = []
        if not isinstance(label_transform, list):
            label_transform = [label_transform]
        self.label_transform = label_transform

        # Labels
        self.return_labels = return_labels
        if return_labels:
            assert labels is None or label_fns is None
            if labels is not None:
                self.labels = np.array(labels)
            else:
                self.labels = ['{}/{}'.format(label_root, lbl) for lbl in label_fns]

        # Others
        assert max_offsync_augm >= 0
        self.max_offsync_augm = max_offsync_augm
        assert clips_per_video > 0
        self.clips_per_video = clips_per_video if sampling == 'clip' else clips_per_video * int(self.num_clips / self.num_videos)
        assert augm_per_clip > 0
        self.augm_per_clip = augm_per_clip
        assert mode in {'video', 'clip'}
        self.mode = mode

        self.shuffle = shuffle
        self.shuffle_dataset()

    def shuffle_dataset(self):
        # Shuffle within this class so that same video is not sampled multiple times in the same batch.
        sample_idx = []
        for _ in range(self.clips_per_video):
            idx = list(range(self.num_samples))
            if self.shuffle:
                random.shuffle(idx)
            sample_idx += idx
        self.sample_idx = np.array(sample_idx)

    def __getitem__(self, index):
        sample_idx = self.sample_idx[index]
        clip_idx = sample_idx if self.sampling == 'clip' else random.sample(self.video2clips[self.video_ids[sample_idx]], 1)[0]

        sample = defaultdict(list)

        video_fn = '{}/{}'.format(self.video_root, self.video_fns[clip_idx].decode())
        if self.mode == 'clip':
            for _ in range(self.augm_per_clip):
                # Clip mode: Randomly samples clips
                video_ts, video_dur = self.random_clip_ts(self.video_lengths[clip_idx])
                clip_data = self.get_clip(clip_idx, video_fn, video_ts, video_dur)
                for k in clip_data:
                    if isinstance(clip_data[k], tuple):
                        sample[k] += list(clip_data[k])
                    else:
                        sample[k] += [clip_data[k]]

            for k in sample:
                if isinstance(sample[k], list) and isinstance(sample[k][0], torch.Tensor):
                    sample[k] = torch.stack(sample[k], 0)
                else:
                    sample[k] = torch.tensor(sample[k])
        else:
            # Video mode: Reads entire video and uniformly splits into self.clips_per_video chunks
            # video_dur = final_time - start_time
            clip_data = self.get_clip(clip_idx, video_fn, 0, self.video_lengths[clip_idx])

            if self.return_video:
                nf = clip_data['video'].shape[1]
                clip_size = int(self.video_clip_duration)
                assert nf >= clip_size
                ts = np.linspace(0, nf-clip_size, self.clips_per_video).astype(int)
                sample['video'] = torch.stack([clip_data['video'][:, ss:ss+clip_size] for ss in ts])

            if self.return_labels:
                # sample['label'] = clip_data['label']
                assert self.return_video
                sample['label'] = torch.stack([clip_data['label'][ss:ss+clip_size] for ss in ts])

        return sample

    def __len__(self):
        if self.mode == 'clip':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Num videos: {}\n - Num clips: {}\n - Epoch size: {}\n".format(
            self.name, self.root, self.num_videos, self.num_clips, self.num_samples * self.clips_per_video)
        if self.return_video:
            desc += " - Example video: {}/{}\n".format(self.video_root, self.video_fns[0].decode())
        return desc

    def get_clip_time_lims(self, video_fn):
        start_time, final_time = [], []
        if self.return_video:
            stream = video_fn.streams.video[0]
            video_dur = stream.duration * stream.time_base
            video_start_time = stream.start_time * stream.time_base
            start_time += [video_start_time]
            final_time += [video_start_time + video_dur]

        start_time = max(start_time)
        final_time = min(final_time)

        return start_time, final_time

    def random_clip_ts(self, duration):
        if self.video_clip_duration > duration:
            return 0, duration
        else:
            clip_ts = random.randint(0, duration - self.video_clip_duration)
            return clip_ts, self.video_clip_duration

    def get_clip(self, clip_idx, video_fn, video_start_time, video_clip_duration):
        sample = {}

        # Load data
        if self.return_video:
            frame_idx = range(video_start_time, video_start_time + video_clip_duration)
            video = [Image.open(video_fn + '/{:05d}.jpg'.format(t)) for t in frame_idx]
        
        if self.return_labels:
            lbl = self.labels[clip_idx]
            if isinstance(lbl, str):
                frame_idx = range(video_start_time, video_start_time + video_clip_duration)
                lbl = [Image.open(lbl + '/{:05d}.png'.format(t)) for t in frame_idx]
            elif isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl)
            else:
                sample['label'] = lbl

        # Transforms
        if self.return_video and self.return_labels:
            for t in self.joint_transform:
                video, lbl = t(video, lbl)

        if self.return_video:
            for t in self.video_transform:
                if isinstance(video, tuple):
                    video = tuple(map(t, video))
                else:
                    video = t(video)
            sample['video'] = video

        if self.return_labels:
            for t in self.label_transform:
                if isinstance(lbl, tuple):
                    lbl = tuple(map(t, lbl))
                else:
                    lbl = t(lbl)
            sample['label'] = lbl

        return sample
