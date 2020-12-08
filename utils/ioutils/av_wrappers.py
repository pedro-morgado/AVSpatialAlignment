import av
import numpy as np
from fractions import Fraction
av.logging.set_level(0)


def av_meta(inpt, video=None, audio=None, format=None):
    if isinstance(inpt, str):
        try:
            container = av.open(inpt, format=format)
        except av.AVError:
            return None, None
    else:
        container = inpt

    if video is not None and len(container.streams.video) > video:
        video_stream = container.streams.video[video]
        time_base = video_stream.time_base
        duration = video_stream.duration * time_base
        start_time = video_stream.start_time * time_base
        width = video_stream.width
        height = video_stream.height
        fps = video_stream.average_rate
        nframes = video_stream.frames
        video_meta = {'fps': fps, 'size': (width, height), 'start_time': start_time, 'duration': duration, 'nframes': nframes}
    else:
        video_meta = None

    if audio is not None and len(container.streams.audio) > audio:
        audio_stream = container.streams.audio[audio]
        time_base = audio_stream.time_base
        duration = audio_stream.duration * time_base
        start_time = audio_stream.start_time * time_base
        channels = audio_stream.channels
        fps = audio_stream.rate
        chunk_size = audio_stream.frame_size
        chunks = audio_stream.frames
        audio_meta = {'channels': channels, 'fps': fps, 'start_time': start_time, 'duration': duration, 'chunks': chunks, 'chunk_size': chunk_size}
    else:
        audio_meta = None

    return video_meta, audio_meta


def av_video_loader(path=None, container=None, rate=None, start_time=None, duration=None):
    # Extract video frames
    if container is None:
        container = av.open(path)
    video_stream = container.streams.video[0]

    # Parse metadata
    _rate = video_stream.average_rate
    _ss = video_stream.start_time * video_stream.time_base
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur

    if rate is None:
        rate = _rate
    if start_time is None:
        start_time = _ss
    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    # Figure out which frames to read in advance
    out_times = [t for t in np.arange(start_time, min(end_time, _ff) - 0.5 / _rate, 1. / rate)]
    out_frame_no = [int((t - _ss) * _rate) for t in out_times][:int(duration * rate)]
    start_time = out_frame_no[0] / float(_rate)

    import time

    # Read data
    n_read = 0
    video = [None] * len(out_frame_no)
    container.seek(int(start_time * av.time_base))

    ts = time.time()
    decode_time, cpu_time = 0., 0.
    for frame in container.decode(video=0):
        decode_time += time.time() - ts
        ts = time.time()
        if n_read == len(out_frame_no):
            break
        frame_no = frame.pts * frame.time_base * _rate
        if frame_no < out_frame_no[n_read]:
            continue
        pil_img = frame.to_image()
        while frame_no >= out_frame_no[n_read]:    # This 'while' takes care of the case where _rate < rate
            video[n_read] = pil_img
            n_read += 1
            if n_read == len(out_frame_no):
                break
        cpu_time += time.time() - ts
        ts = time.time()
    video = [v for v in video if v is not None]

    return (video, rate), (decode_time, cpu_time)

def av_audio_loader(path=None, container=None, rate=None, start_time=None, duration=None, layout="4.0"):
    if container is None:
        container = av.open(path)
    audio_stream = container.streams.audio[0]

    # Parse metadata
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _rate = audio_stream.rate

    if rate is None:
        rate = _rate
    if start_time is None:
        start_time = _ss
    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=rate)

    # Read data
    chunks = []
    container.seek(int(start_time * av.time_base))
    for frame in container.decode(audio=0):
        chunk_start_time = frame.pts * frame.time_base
        chunk_end_time = chunk_start_time + Fraction(frame.samples, frame.rate)
        if chunk_end_time < start_time:   # Skip until start time
            continue
        if chunk_start_time > end_time:       # Exit if clip has been extracted
            break

        try:
            frame.pts = None
            if resampler is not None:
                chunks.append((chunk_start_time, resampler.resample(frame).to_ndarray()))
            else:
                chunks.append((chunk_start_time, frame.to_ndarray()))
        except AttributeError:
            break

    # Trim for frame accuracy
    audio = np.concatenate([af[1] for af in chunks], 1)
    ss = int((start_time - chunks[0][0]) * rate)
    t = int(duration * rate)
    if ss < 0:
        audio = np.pad(audio, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    audio = audio[:, ss: ss+t]

    # Normalize to [-1, 1]
    audio = audio / np.iinfo(audio.dtype).max

    return audio, rate


def av_load_all_video(path, video_fps=None):
    container = av.open(path)

    if len(container.streams.video)==0:
        return None

    # Extract video frames
    video_stream = container.streams.video[0]
    vid_ss = video_stream.start_time * video_stream.time_base
    vid_dur = video_stream.duration * video_stream.time_base
    vid_ff = vid_ss + vid_dur
    vid_fps = video_stream.average_rate

    if video_fps is None:
        video_fps = vid_fps

    outp_times = np.arange(vid_ss, vid_ff, 1./video_fps)
    outp_times = [t for t in outp_times if t < vid_ff]
    outp_vframes = [int((t - vid_ss) * vid_fps) for t in outp_times]

    vframes = []
    container.seek(int(outp_times[len(vframes)] * av.time_base))
    for frame in container.decode(video=0):
        if len(vframes) == len(outp_vframes):
            break
        frame_id = frame.pts * frame.time_base * video_stream.rate
        if frame_id < outp_vframes[len(vframes)]:
            continue
        pil_img = frame.to_image()
        while frame_id >= outp_vframes[len(vframes)]:
            vframes.append((frame.time, pil_img))
            if len(vframes) == len(outp_vframes):
                break

    vframes = [vf[1] for vf in vframes]
    return vframes, video_fps


def av_load_all_audio(path, audio_fps=None):
    container = av.open(path)

    if len(container.streams.audio) == 0:
        return None

    audio_stream = container.streams.audio[0]
    snd_fps = audio_stream.rate

    if audio_fps is None:
        resample = False
        audio_fps = snd_fps
    else:
        resample = True
        audio_resampler = av.audio.resampler.AudioResampler(format="s16p", layout="mono", rate=audio_fps)

    aframes = []
    for frame in container.decode(audio=0):
        frame_pts = frame.pts * frame.time_base
        if resample:
            np_snd = audio_resampler.resample(frame).to_ndarray()
        else:
            np_snd = frame.to_ndarray()
        aframes.append((frame_pts, np_snd))

    aframes = np.concatenate([af[1] for af in aframes], 1)
    aframes = aframes / np.iinfo(aframes.dtype).max

    return aframes, audio_fps


def av_write_video(fn, video_frames, fps):
    assert fn.endswith('.mp4')

    container = av.open(fn, mode='w')
    stream = container.add_stream('mpeg4', rate=fps)
    stream.width = video_frames.shape[2]
    stream.height = video_frames.shape[1]
    # stream.pix_fmt = 'yuv420p'
    stream.pix_fmt = 'gray'

    for img in video_frames:
        img = np.round(img).astype(np.uint8)
        img = np.clip(img, 0, 255)

        # frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        frame = av.VideoFrame.from_ndarray(img, format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


if __name__ == '__main__':
    av_load_all_video('/datasets01_101/ucf101/112018/data/v_ApplyLipstick_g01_c01.avi', video_fps=8,)
    av_load_all_audio('/datasets01_101/ucf101/112018/data/v_ApplyLipstick_g01_c01.avi', audio_fps=48000, )
