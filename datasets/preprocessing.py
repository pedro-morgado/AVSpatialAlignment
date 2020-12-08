import torch
import numpy as np
import random
import librosa
import os
import math
from utils.video360 import AmbiPower
from PIL import Image
from scipy.spatial.transform import Rotation as R


class VideoPadMissing(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, frames):
        if len(frames) == self.num_frames:
            return frames
        else:
            return [frames[i%len(frames)] for i in range(self.num_frames)]



class VideoPrep_MSC_CJ(object):
    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 min_area=0.08,
                 augment=True,
                 num_frames=8,
                 pad_missing=False,
                 ):
        from utils.videotransforms import video_transforms, volume_transforms, tensor_transforms
        self.crop = crop
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing

        if augment:
            transforms = [
                video_transforms.RandomResizedCrop(crop, scale=(min_area, 1.)),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
        else:
            transforms = [
                video_transforms.Resize(int(crop[0]/0.875)),
                video_transforms.CenterCrop(crop),
            ]

        if pad_missing:
            transforms += [VideoPadMissing(num_frames)]

        transforms += [volume_transforms.ClipToTensor()]
        transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        return self.transform(frames)


class VideoPrep_CJ(object):
    def __init__(self,
                 color=(0.4, 0.4, 0.4, 0.2),
                 augment=True,
                 num_frames=8,
                 pad_missing=False,
                 random_flip=True,
                 random_color=True,
                 ):
        from utils.videotransforms import video_transforms, volume_transforms, tensor_transforms
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing

        transforms = []
        if augment:
            if random_flip:
                transforms += [video_transforms.RandomHorizontalFlip()]
            if random_color:
                transforms += [video_transforms.ColorJitter(*color)]

        if pad_missing:
            transforms += [VideoPadMissing(num_frames)]

        transforms += [volume_transforms.ClipToTensor()]
        transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        return self.transform(frames)


class AudioPrep(object):
    def __init__(self, duration, mono=False, volume=0.1, augment=False):
        self.augment = augment
        self.volume = volume
        self.mono = mono
        self.duration = duration

    def __call__(self, sig, sr, duration=None):
        if duration is None:
            duration = self.duration
        num_frames = int(duration * sr)

        # Downmix to mono
        if self.mono:
            sig = sig.mean(0, keepdims=True).astype(np.float32) if sig.shape[0] <= 2 else sig[:1].astype(np.float32)

        # Trim or pad to constant shape
        if sig.shape[1] > num_frames:
            sig = sig[:, :num_frames]
        elif sig.shape[1] < num_frames:
            n_pad = num_frames - sig.shape[1]
            sig = np.pad(sig, ((0, 0), (0, n_pad)), mode='constant', constant_values=0.)

        # Augment by randomly adjust volume
        if self.augment:
            sig *= random.uniform(1.-self.volume, 1.+self.volume)
            sig[sig > 1.] = 1.
            sig[sig < -1.] = -1.

        # sig = torch.from_numpy(sig)
        return sig, sr


class LogMelSpectrogram(object):
    def __init__(self, fps, n_mels=128, n_fft=2048, hop_size=0.005, normalize=False):
        self.inp_fps = fps
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.rate = 1./hop_size
        self.normalize = normalize

        if self.normalize:
            fn = f'datasets/assets/MelSpectDB-{fps // 1000}khz-{n_fft // 2 + 1}fft-{n_mels}mels-norm-stats.npz'
            if not os.path.isfile(fn):
                compute_mel_stats(fps, n_mels=n_mels, n_fft=n_fft, fn=fn)
            stats = np.load(fn)
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, sig, sr, duration=None):
        hop_length = int(self.hop_size * sr)
        mels_spect = []
        for y in sig:
            y = np.asfortranarray(y)  # Fortran contiguity required by librosa
            mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=hop_length, power=1.0)
            if duration is not None:
                mels = mels[:, :int(duration * self.rate)]

            mels = librosa.core.power_to_db(mels, top_db=100)
            if self.normalize:
                mels = (mels - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-5)

            mels_spect += [mels]
        mels_t = torch.stack([torch.from_numpy(mels) for mels in mels_spect]) # Channels x Freq x Time
        mels_t = torch.transpose(mels_t, 1, 2)  # Channels x Time x Freq
        return mels_t, self.rate


def compute_mel_stats(fps, n_mels, n_fft, fn):
    from datasets import YT360
    import torch.utils.data as data

    audio_transform = [
        AudioPrep(duration=2.0, augment=True),
        LogMelSpectrogram(fps, n_fft=n_fft, n_mels=n_mels, hop_size=1. / 64., normalize=False, mode=mode)
    ]
    dataset = YT360(
        return_video=False,
        return_audio=True,
        audio_clip_duration=2.0,
        audio_fps=fps,
        spect_fps=64.,
        audio_transform=audio_transform,
        clips_per_video=1,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    audio = []
    for batch in loader:
        audio += [batch['audio'].numpy()]
        if len(audio) % 10 == 0:
            print(len(audio), len(loader))
        if len(audio) > 1000:
            break
    audio = np.concatenate(audio, 0)
    audio = audio.reshape(-1, audio.shape[-1])
    mu = audio.mean(0)
    std = audio.std(0)
    np.savez(fn, mean=mu, std=std)


class Crop360Video(object):
    def __init__(self, height, width, audio_input='mono'):
        from utils.nfov import NFOV
        self.nfov = NFOV(height=height, width=width)
        self.audio_input = audio_input

    def __call__(self, video, audio, segm, center, fov):
        from utils.ambisonics.decoder import decode_ambix
        from utils.ambisonics.position import Position
        from utils.video360 import rotate_ambix
        from utils.video360 import er2polar
        from utils.ambisonics.binauralizer import DirectAmbisonicBinauralizer, AmbisonicDecoder
        from utils.ambisonics.common import AmbiFormat

        # Crop frames
        self.nfov.setFOV(fov)
        self.nfov.setCenterPoit(center)
        video = [Image.fromarray(self.nfov.toNFOV(np.asarray(img))) for img in video]

        # Crop segmentation maps
        if segm is not None:
            segm = self.nfov.toNFOV(segm, interpolation='nn')

        # Decode ambisonics into center point direction
        pos_ambix = Position(*er2polar(*center), 'polar')
        if self.audio_input == 'mono':
            audio = decode_ambix(audio, pos_ambix).astype(np.float32)
        elif self.audio_input == 'stereo':
            audio = rotate_ambix(audio, pos_ambix)[0].astype(np.float32)
            audio = DirectAmbisonicBinauralizer(AmbiFormat(), method='projection').binauralize(audio)
        else:
            audio = rotate_ambix(audio, pos_ambix)[0].astype(np.float32)
        return video, audio, segm


class SpatialVideoCropTool(object):
    def __init__(self,
                 size=(112, 112),
                 hfov_lims=(0.0707, 0.25),
                 ratio=(3. / 4., 4. / 3.),
                 horizon_only=False,
                 random_flip=False,
                 margin=0.,
                 audio_input='ambix',
                 pos='random',
                 num_crops=1,
                 res=0.1,
                 alpha=1,):
        self.crop_tool = Crop360Video(height=size[0], width=size[1], audio_input=audio_input)
        self.hfov_lims = hfov_lims
        self.ratio = ratio
        self.horizon_only = horizon_only
        self.margin = margin
        self.pos = pos
        self.audio_input = audio_input
        self.num_crops = num_crops
        self.random_flip = random_flip
        self.rotation = None

        if pos == 'ambi_pow':
            self.pos_res = res
            self.pos_list = self._pos_grid(res=res)
            self.ambi_pow = AmbiPower(self.pos_list)
            self.alpha = alpha

    def _pos_grid(self, res):
        return [(x, y) for x in np.arange(0, 1, res) for y in np.arange(0, 1, res)]

    def _sample_fovs(self, img, hfov_lims, ratio):
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        fov = []
        for _ in range(self.num_crops):
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            hf = random.uniform(*hfov_lims)
            vf = aspect_ratio * hf * img.size[0] / img.size[1]
            fov += [(hf, vf)]
        return np.array(fov)

    def _sample_center_points(self, frames, sig):
        if self.pos == 'random':
            if self.horizon_only:
                return np.array([[random.uniform(0., 1.), 0.5] for _ in range(self.num_crops)])
            else:
                return np.array([[random.uniform(0., 1.), random.uniform(0.33, 0.66)] for _ in range(self.num_crops)])
        elif self.pos == 'ambi_pow':
            pow_map = self.ambi_pow.compute(sig)
            if pow_map.sum() < 1e-4:
                raise Exception('no audio')
            weight = pow_map ** self.alpha
            pos_idx = np.random.choice(len(self.pos_list), size=1, replace=False, p=weight / weight.sum())
            return np.array([self.pos_list[i] for i in pos_idx])
        else:
            raise NotImplementedError

    def __call__(self, inp_video, inp_audio, inp_segm=None, return_inv=False, return_rotation=False):
        # Misalign
        if self.rotation is not None:
            if isinstance(self.rotation, list):
                inp_audio = np.repeat(inp_audio[..., None], len(self.rotation), axis=-1)
                for i, rot in enumerate(self.rotation):
                    if rot is not None:
                        inp_audio[1:, ..., i] = rot.apply(inp_audio[1:, ..., i].T).T
            else:
                inp_audio[1:] = self.rotation.apply(inp_audio[1:].T).T

        # Random flip
        if self.random_flip:
            if random.random() > 0.5:
                inp_video = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in inp_video]
                if inp_segm is not None:
                    inp_segm = inp_segm[:, ::-1]
                if inp_audio.shape[0] == 4: # Ambisonics
                    inp_audio[1, :] *= -1

        # Crop
        video_crops, audio_crops, segm_crops, inv_map = [], [], [], []
        for cp, fov in zip(self.center_points, self.fovs):
            video, audio, segm = self.crop_tool(inp_video, inp_audio, inp_segm, cp, fov)
            video_crops += [video]
            audio_crops += [audio]
            segm_crops += [segm]

        if self.rotation is not None:
            if isinstance(self.rotation, list):
                rotation = torch.stack([torch.from_numpy(rot.as_quat()[[3, 0, 1, 2]]).float() if rot is not None else torch.zeros(4) for rot in self.rotation])
            else:
                rotation = torch.from_numpy(self.rotation.as_quat()[[3, 0, 1, 2]]).float()  # xyzw -> wxyz
        else:
            rotation = torch.zeros(4)
        
        if return_rotation:
            return video_crops, audio_crops, segm_crops, torch.from_numpy(self.center_points), rotation

        return video_crops, audio_crops, segm_crops, torch.from_numpy(self.center_points)

    def reset_crop_hist(self):
        self.center_hist = np.zeros((0, 2))

    def add_crop_to_hist(self, crop):
            self.center_hist = np.concatenate((self.center_hist, crop * [2., 1.]), 0)
            self.center_hist = np.concatenate((self.center_hist, crop * [2., 1.] + [2., 0.]), 0)
            self.center_hist = np.concatenate((self.center_hist, crop * [2., 1.] + [-2., 0.]), 0)

    def sample_rotations(self, p=0, mode='quat'):
        if random.random() < p:
            if mode == 'quat':
                self.rotation = R.from_quat(torch.randn(4))
            elif mode == 'yaw':
                yaw = random.uniform(-math.pi, math.pi)
                self.rotation = R.from_euler('y', yaw)
            elif mode == 'pitch-yaw':
                pitch = random.uniform(-math.pi/2, math.pi/2)
                yaw = random.uniform(-math.pi, math.pi)
                self.rotation = R.from_euler('xy', [pitch, yaw])
            elif mode == 'yaw-pitch':
                pitch = random.uniform(-math.pi/2, math.pi/2)
                yaw = random.uniform(-math.pi, math.pi)
                self.rotation = R.from_euler('yx', [yaw, pitch])
            else:
                raise Exception('rotation mode not supported:' + mode)
        else:
            self.rotation = None

    def sample_crops(self, inp_video, inp_audio):
        while True:
            # Sample crop orientation and scale
            self.center_points = self._sample_center_points(inp_video, inp_audio)
            self.fovs = self._sample_fovs(inp_video[0], self.hfov_lims, self.ratio)

            if self.center_hist.shape[0] == 0:
                self.add_crop_to_hist(self.center_points)
                break
            else:
                closest_center_dist = ((self.center_points * [2., 1.] - self.center_hist) ** 2).sum(1).min() ** 0.5
                if closest_center_dist > self.margin:
                    self.add_crop_to_hist(self.center_points)
                    break


class Spatial2Planar(object):
    def __init__(self, size=(256, 256)):
        from utils.nfov import NFOV
        self.nfov = NFOV(height=size[0], width=size[1])
        self.nfov.setFOV([0.25, 0.5])

    def __call__(self, frames, sig):
        from utils.ambisonics.decoder import decode_ambix
        from utils.ambisonics.position import Position
        from utils.video360 import er2polar

        # Crop frames
        center_point = np.array([random.uniform(0., 1.), random.uniform(0.33, 0.66)])
        self.nfov.setCenterPoit(center_point)
        video = [Image.fromarray(self.nfov.toNFOV(np.asarray(img))) for img in frames]

        # Decode ambisonics into center point direction
        pos_ambix = Position(*er2polar(*center_point), 'polar')
        audio = decode_ambix(sig, pos_ambix).astype(np.float32)

        return video, audio


class VideoLabelPrep_Segm(object):
    def __init__(self,
                 resize=(256, 256),
                 crop=(224, 224),
                 augment=True
                 ):
        from utils.videotransforms import video_transforms
        self.resize = resize
        self.crop = crop
        self.augment = augment

        if augment:
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.RandomCrop(crop),
                video_transforms.RandomHorizontalFlip(),
            ]
        else:
            transforms = [
                video_transforms.Resize(crop),
            ]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames, labels):
        assert isinstance(frames, list) and isinstance(labels, list)
        joint_input = frames + labels
        joint_output = self.transform(joint_input)
        return joint_output[:len(frames)], joint_output[len(frames):]


class VideoPrep_Segm(object):
    def __init__(self,
                 color=(0.4, 0.4, 0.4, 0.2),
                 augment=True,
                 num_frames=8,
                 pad_missing=False,
                 normalize=True
                 ):
        from utils.videotransforms import video_transforms, volume_transforms, tensor_transforms
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing

        transforms = []
        if augment:
            transforms += [video_transforms.ColorJitter(*color)]

        if pad_missing:
            transforms += [VideoPadMissing(num_frames)]

        transforms += [volume_transforms.ClipToTensor()]
        if normalize:
            transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        return self.transform(frames)