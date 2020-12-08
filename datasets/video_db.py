import random
import torch
import numpy as np
import torch.utils.data as data
from utils.ioutils import av_wrappers
from collections import defaultdict
import av
from PIL import Image

import math

def chararray(fn_list):
    charr = np.chararray(len(fn_list), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]
    return charr


def load_image(fn):
    return Image.open(fn).convert('RGB')


class VideoDataset(data.Dataset):
    def __init__(self,
                 sampling='video',
                 video2clips=None,
                 return_video=True,
                 video_root=None,
                 video_fns=None,
                 video_clip_duration=1.,
                 video_fps=25,
                 return_audio=True,
                 audio_root=None,
                 audio_fns=None,
                 use_ambix=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 spect_fps=None,
                 return_segmentation=False,
                 segmentation_root=None,
                 segmentation_dirs=None,
                 joint_transform=None,
                 video_transform=None,
                 audio_transform=None,
                 segmentation_transform=None,
                 return_position=False,
                 return_labels=False,
                 labels=None,
                 mode='clip',
                 augm_per_clip=1,
                 use_temporal_augm=True,
                 use_spatial_augm=False,
                 misalign=0,
                 rotate_mode='quat',
                 clips_per_video=1,
                 max_offsync_augm=0,
                 shuffle=True,
                 ):
        super(VideoDataset, self).__init__()

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
        self.video_fps = video_fps

        # Audio config
        self.return_audio = return_audio
        self.use_ambix = use_ambix
        self.audio_root = audio_root
        assert audio_clip_duration > 0
        self.audio_clip_duration = audio_clip_duration
        if return_audio:
            self.audio_fns = chararray(audio_fns)
        self.audio_fps = audio_fps
        self.spect_fps = spect_fps

        # Segmentation config
        self.return_segmentation = return_segmentation
        self.segmentation_root = segmentation_root
        if return_segmentation:
            self.segmentation_dirs = chararray(segmentation_dirs)
        self.segmentation_transform = segmentation_transform

        # Transforms
        self.joint_transform = joint_transform
        if joint_transform is not None:
            assert self.return_video and self.return_audio

        if video_transform is None:
            video_transform = []
        if not isinstance(video_transform, list):
            video_transform = [video_transform]
        self.video_transform = video_transform

        if audio_transform is None:
            audio_transform = []
        if not isinstance(audio_transform, list):
            audio_transform = [audio_transform]
        self.audio_transform = audio_transform


        # Labels
        self.return_labels = return_labels
        if return_labels:
            self.labels = np.array(labels)

        # Others
        self.return_position = return_position
        assert max_offsync_augm >= 0
        self.max_offsync_augm = max_offsync_augm
        assert clips_per_video > 0
        self.clips_per_video = clips_per_video if sampling == 'clip' else clips_per_video * int(self.num_clips / self.num_videos)
        assert augm_per_clip > 0
        self.augm_per_clip = augm_per_clip
        self.use_temporal_augm = use_temporal_augm
        self.use_spatial_augm = use_spatial_augm
        assert mode in {'video', 'clip'}
        self.mode = mode

        self.misalign = misalign
        self.rotate_mode = rotate_mode

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

    def getitem(self, index, position=None):
        sample_idx = self.sample_idx[index]
        clip_idx = sample_idx if self.sampling == 'clip' else random.sample(self.video2clips[self.video_ids[sample_idx]], 1)[0]

        sample = defaultdict(list)

        video_ctr = audio_ctr = segm_dir = None
        if self.return_video:
            video_fn = '{}/{}'.format(self.video_root, self.video_fns[clip_idx].decode())
            video_ctr = av.open(video_fn)
        if self.return_audio:
            audio_fn = '{}/{}'.format(self.audio_root, self.audio_fns[clip_idx].decode())
            audio_ctr = av.open(audio_fn)
        if self.return_segmentation:
            segm_dir = '{}/{}'.format(self.segmentation_root, self.segmentation_dirs[clip_idx].decode())

        start_time, final_time = self.get_clip_time_lims(video_ctr, audio_ctr)
        if self.mode == 'clip':
            # Clip mode: Randomly samples clips
            clips_t = self.sample_clips(start_time, final_time)
            clip_data = self.get_clips(clip_idx, video_ctr, audio_ctr, segm_dir, clips_t)
            sample.update(clip_data)

        else:
            # Video mode: Reads entire video and uniformly splits into self.clips_per_video chunks
            duration = final_time - start_time
            clip_data = self.get_clips(clip_idx, video_ctr, audio_ctr, segm_dir, [(start_time, duration, start_time, duration)])

            if self.return_video:
                nf = clip_data['video'].shape[2]
                clip_size = int(self.video_clip_duration * self.video_fps)
                assert nf >= clip_size
                ts = np.linspace(0, nf-clip_size, self.clips_per_video).astype(int)
                sample['video'] = torch.stack([clip_data['video'][:, :, ss:ss+clip_size] for ss in ts], 1)

            if self.return_audio:
                nf = clip_data['audio'].shape[2]
                clip_size = int(self.audio_clip_duration * self.spect_fps)
                assert nf >= clip_size
                ts = np.linspace(0, nf - clip_size, self.clips_per_video).astype(int)
                sample['audio'] = torch.stack([clip_data['audio'][:, :, ss:ss+clip_size] for ss in ts], 1)

            if self.return_labels:
                sample['label'] = clip_data['label']

        if self.return_video:
            try:
                video_ctr.close()
            except AttributeError:
                del video_ctr
        if self.return_audio:
            try:
                audio_ctr.close()
            except AttributeError:
                del audio_ctr

        return sample

    def __getitem__(self, index):
        return self.getitem(index)

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
        if self.return_audio:
            desc += " - Example audio: {}/{}\n".format(self.audio_root, self.audio_fns[0].decode())
        return desc

    def get_clip_time_lims(self, video_ctr, audio_ctr):
        start_time, final_time = [], []
        if self.return_video:
            stream = video_ctr.streams.video[0]
            video_dur = stream.duration * stream.time_base
            video_start_time = stream.start_time * stream.time_base
            start_time += [video_start_time]
            final_time += [video_start_time + video_dur]

        if self.return_audio:
            stream = audio_ctr.streams.audio[0]
            audio_dur = stream.duration * stream.time_base
            audio_ss = stream.start_time * stream.time_base
            start_time += [audio_ss]
            final_time += [audio_ss + audio_dur]

        start_time = max(start_time)
        final_time = min(final_time)

        return start_time, final_time

    def sample_clips(self, clip_start_time, clip_final_time):
        if self.use_temporal_augm:
            # clips_t = []
            # for _ in range(self.augm_per_clip):
            #     clips_t += [self.random_clip_ts(clip_start_time, clip_final_time)]
            clips_t = []
            chunk_size = (clip_final_time - clip_start_time) / self.augm_per_clip
            for i in range(self.augm_per_clip):
                st = chunk_size * i
                ft = chunk_size * (i + 1)
                clips_t += [self.random_clip_ts(st, ft)]
        else:
            video_ts, video_dur, audio_ts, audio_dur = self.random_clip_ts(clip_start_time, clip_final_time)
            clips_t = [(video_ts, video_dur, audio_ts, audio_dur) for _ in range(self.augm_per_clip)]
        return clips_t

    def random_clip_ts(self, clip_start_time, clip_final_time):
        duration = clip_final_time - clip_start_time

        if not self.return_audio:
            if self.video_clip_duration > duration:
                return clip_start_time, duration, clip_start_time, duration
            else:
                clip_ts = random.uniform(clip_start_time, clip_final_time - self.video_clip_duration)
                return clip_ts, self.video_clip_duration, clip_ts, self.video_clip_duration

        elif not self.return_video:
            if self.audio_clip_duration > duration:
                return clip_start_time, duration, clip_start_time, duration
            else:
                clip_ts = random.uniform(clip_start_time, clip_final_time - self.audio_clip_duration)
                return clip_ts, self.audio_clip_duration, clip_ts, self.audio_clip_duration

        else:
            # Return both video and audio
            min_ss = clip_start_time
            max_ss = min(clip_final_time - self.audio_clip_duration, clip_final_time - self.video_clip_duration)
            assert max_ss > min_ss
            if self.audio_clip_duration > self.video_clip_duration:
                # Sample audio first, then video within a max_offsync_augm window
                audio_ts = random.uniform(min_ss, max_ss)

                win_min = max(audio_ts - self.max_offsync_augm, clip_start_time)
                win_max = min(audio_ts + self.audio_clip_duration + self.max_offsync_augm, clip_final_time) - self.video_clip_duration
                video_ts = random.uniform(win_min, win_max)
                return video_ts, self.video_clip_duration, audio_ts, self.audio_clip_duration
            else:
                # Sample video first, then audio within a max_offsync_augm window
                video_ts = random.uniform(min_ss, max_ss)

                win_min = max(video_ts - self.max_offsync_augm, clip_start_time)
                win_max = min(video_ts + self.video_clip_duration + self.max_offsync_augm, clip_final_time) - self.audio_clip_duration
                audio_ts = random.uniform(win_min, win_max)
                return video_ts, self.video_clip_duration, audio_ts, self.audio_clip_duration

    def get_clips(self, clip_idx, video_ctr, audio_ctr, segm_dir, clips_t):
        sample = defaultdict(list)

        # Load data
        video, audio, segm = [], [], []
        for video_st, video_dur, audio_st, audio_dur in clips_t:
            if self.return_video:
                video_dt, video_fps = self.load_video_clip(video_ctr, video_st, video_dur)
                video += [video_dt]
            if self.return_audio:
                audio_dt, audio_fps = self.load_audio_clip(audio_ctr, audio_st, audio_dur)
                audio += [audio_dt]
            if self.return_segmentation:
                segm_dt = self.load_segm_clip(segm_dir, video_st, video_dur)
                segm += [segm_dt]
        if not self.return_segmentation:
            segm = None

        # Transforms
        if self.joint_transform is not None:
            if self.misalign > 0:
                video_clips, audio_clips, segm_clips, positions, rotations = self.apply_joint_transform(video, audio, clips_t, segm)
            else:
                video_clips, audio_clips, segm_clips, positions = self.apply_joint_transform(video, audio, clips_t, segm)
        else:
            video_clips, audio_clips, segm_clips = video, audio, segm

        if self.return_video:
            sample['video'] = self.apply_video_transform(video_clips)
        if self.return_audio:
            sample['audio'] = self.apply_audio_transform(audio_clips, audio_fps, audio_dur)
        if self.return_segmentation:
            sample['segmentation'] = torch.stack([torch.from_numpy(self.segmentation_transform(segm)) for segm in segm_clips], 0)
        if self.return_position:
            sample['position'] = positions
        if self.misalign > 0:
            sample['rotation'] = rotations

        # Labels
        if self.return_labels:
            lbl = self.labels[clip_idx]
            sample['label'] = torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl

        return sample

    def load_video_clip(self, ctr, start_time, duration):
        (video_dt, video_fps), _ = av_wrappers.av_video_loader(
            container=ctr,
            rate=self.video_fps,
            start_time=start_time,
            duration=duration,
        )
        return video_dt, video_fps

    def load_audio_clip(self, ctr, start_time, duration):
        audio_dt, audio_fps = av_wrappers.av_audio_loader(
            container=ctr,
            rate=self.audio_fps,
            start_time=start_time,
            duration=duration,
            layout='4.0' if self.use_ambix else 'mono'
        )
        return audio_dt, audio_fps

    def load_segm_clip(self, segm_dir, start_time, duration):
        segm_frame_no = int((start_time + duration) * 4)
        for n in reversed(range(0, segm_frame_no+1)):
            try:
                things_segm = np.array(load_image(f"{segm_dir}/things-{n:04d}.png"))[:, :, 0]
                stuff_segm = np.array(load_image(f"{segm_dir}/stuff-{n:04d}.png"))[:, :, 0]
                break
            except FileNotFoundError:
                continue
        if n == 0:
            raise Exception(f"segm maps not found at {segm_dir}")
        num_things, num_stuff = 80, 54
        segm = things_segm
        segm[segm == num_things] = stuff_segm[segm == num_things] + num_things
        segm[segm == num_things+num_stuff] = 255
        return segm

    def apply_joint_transform(self, video_clips, audio_clips, clips_t, segmentation_clips=None):
        video_crops_out, audio_crops_out, segm_crops_out, positions_out = [], [], [], []
        self.joint_transform.reset_crop_hist()
        if self.misalign > 0:
            self.joint_transform.sample_rotations(p=self.misalign, mode=self.rotate_mode)
        for i, (video, audio, ts) in enumerate(zip(video_clips, audio_clips, clips_t)):
            # When using spatial augmentation, sample different views for each clip.
            # Otherwise sample the same view for all clips.
            if self.use_spatial_augm or i == 0:
                self.joint_transform.sample_crops(video, audio)
            segm = segmentation_clips[i] if segmentation_clips is not None else None
            if self.misalign > 0:
                video_crops, audio_crops, segm_crops, positions, rotations = self.joint_transform(video, audio, segm, return_rotation=True)
            else:
                video_crops, audio_crops, segm_crops, positions = self.joint_transform(video, audio, segm)
            video_crops_out += video_crops
            audio_crops_out += audio_crops
            segm_crops_out += segm_crops
            ts = (ts[0] + ts[1] / 2. + ts[2] + ts[3] / 2.) / 2. * torch.ones((len(video_crops), 1)) / 10.
            positions_out += [torch.cat((positions.float(), ts.float()), 1)]
        positions_out = torch.cat(positions_out, 0)

        if self.misalign > 0:
            return video_crops_out, audio_crops_out, segm_crops_out, positions_out, rotations
    
        return video_crops_out, audio_crops_out, segm_crops_out, positions_out

    def apply_video_transform(self, video_clips):
        num_clips = len(video_clips)
        video_clips_out = []
        for i in range(num_clips):
            clip = video_clips[i]
            for t in self.video_transform:
                clip = t(clip)
            video_clips_out.append(clip)
        if isinstance(video_clips_out[0], torch.Tensor):
            video_clips_out = torch.stack(video_clips_out, 0)
        return video_clips_out

    def apply_audio_transform(self, audio_clips, audio_fps, audio_dur):
        num_clips = len(audio_clips)
        audio_clips_out = []
        for i in range(num_clips):
            clip, rate = audio_clips[i], audio_fps
            for t in self.audio_transform:
                clip, rate = t(clip, rate, audio_dur)
            audio_clips_out.append(clip)
        if isinstance(audio_clips_out[0], torch.Tensor):
            audio_clips_out = torch.stack(audio_clips_out, 0)
        else:
            audio_clips_out = np.stack(audio_clips_out, 0)
        return audio_clips_out

