import av
import numpy as np
from fractions import Fraction


class AudioReader:
    def __init__(self, video_fn):
        self.container = av.open(video_fn)
        self.stream = self.container.streams.audio[0]

        self._fps = self.stream.rate
        if self.stream.duration is not None:
            self._duration = self.stream.duration * self.stream.time_base
        else:
            self._duration = self.container.duration / av.time_base
        if self.stream.start_time is not None:
            self._start_time = self.stream.start_time * self.stream.time_base
        else:
            self._start_time = self.container.start_time / av.time_base
        self._end_time = self._start_time + self._duration

    def fast_seek(self, seek):
        self.container.seek(int(seek * av.time_base))

    def _read(self, start_time=None, duration=None, target_fps=None):
        start_time = self._start_time if start_time is None else start_time
        duration = self._duration if duration is None else duration
        end_time = min(start_time + duration, self._end_time)
        duration = end_time - start_time

        target_fps = self._fps if target_fps is None else target_fps
        resampler = None
        if target_fps != self._fps:
            resampler = av.audio.resampler.AudioResampler(format="s16p", rate=target_fps)
        total_frames = int(duration * target_fps)

        self.fast_seek(start_time)
        n_read = 0
        for frame in self.container.decode(audio=0):
            frame_pts = frame.pts * frame.time_base
            frame_end_pts = frame_pts + Fraction(frame.samples, frame.rate)
            if frame_end_pts < start_time:   # Skip until start time
                continue
            if frame_pts > end_time:       # Exit if clip has been extracted
                break

            # Resample
            frame.pts = None
            if resampler is not None:
                frame_np = resampler.resample(frame).to_ndarray()
            else:
                frame_np = frame.to_ndarray()

            # Trim first frame
            if n_read == 0:
                ss = int((start_time - frame_pts) * target_fps)
                if ss < 0:  # Requested data before audio starting time
                    frame_np = np.pad(frame_np, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
                else:
                    frame_np = frame_np[:, ss:]

            # Trim last frame
            if n_read + frame_np.shape[1] > total_frames:
                frame_np = frame_np[:, :total_frames-n_read]

            # Normalize to [-1,1]
            if frame_np.dtype != np.float32:
                frame_np = frame_np / np.iinfo(frame_np.dtype).max

            # Return frames
            yield frame_np
            n_read += frame_np.shape[1]

    def read(self, start_time=None, duration=None, target_fps=None):
        data = []
        for frame in self._read(start_time=start_time, duration=duration, target_fps=target_fps):
            data.append(frame)
        data = np.concatenate(data, 1)
        return data


class AudioWriter:
    def __init__(self, fn, fps):
        self.container = av.open(fn, mode='w')
        self.stream = self.container.add_stream('aac')
        self.stream.codec_context.format = 'fltp'
        self.stream.codec_context.layout = '4.0'
        self.stream.codec_context.sample_rate = fps
        # self.stream.codec_context.frame_size = 1024
        self.fps = fps

    def write(self, data):
        # Convert to AudioFrames, encode and write to container
        frame = av.AudioFrame.from_ndarray(data.astype(np.float32), format='fltp', layout='4.0')
        frame.pts = None
        frame.rate = self.fps
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()


if __name__ == '__main__':
    a_reader = AudioReader('scraping/data/prep-final/audio/bb5eETSspVI-213.m4a')
    audio = a_reader.read(start_time=6., duration=4.0, target_fps=24000)
    a_writer = AudioWriter('tmp.m4a', 24000)
    a_writer.write(audio)
