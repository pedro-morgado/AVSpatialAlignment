import av
import numpy as np


class VideoReader:
    def __init__(self, video_fn):
        self.container = av.open(video_fn)
        self.stream = self.container.streams.video[0]

        self.width = self.stream.width
        self.height = self.stream.height
        self._frames = self.stream.frames
        self._fps = self.stream.average_rate
        self._duration = self.stream.duration * self.stream.time_base
        self._start_time = self.stream.start_time * self.stream.time_base
        self._end_time = self._start_time + self._duration

    def fast_seek(self, seek):
        self.container.seek(int(seek * av.time_base))

    def _read(self, start_time=None, duration=None, target_fps=None):
        target_fps = self._fps if target_fps is None else target_fps
        start_time = self._start_time if start_time is None else start_time
        duration = self._duration if duration is None else duration
        target_times = [t for t in np.arange(start_time, start_time + duration - 0.5/target_fps, 1./target_fps)]
        target_frames = [int((t - self._start_time) * self._fps) for t in target_times]

        self.fast_seek(start_time)
        n_read = 0
        for frame in self.container.decode(video=0):
            frame_id = frame.pts * frame.time_base * self._fps
            if frame_id < target_frames[n_read]:
                continue
            pil_img = frame.to_image()
            while frame_id >= target_frames[n_read]:   # Repeat same image in case requested fps > video fps
                yield pil_img
                n_read += 1
                if n_read == len(target_frames):
                    break
            if n_read == len(target_frames):
                break

    def read(self, start_time=None, duration=None, target_fps=None):
        frames = []
        for img in self._read(start_time=start_time, duration=duration, target_fps=target_fps):
            frames.append(img)
        return frames


class VideoWriter:
    def __init__(self, fn, fps, width, height):
        self.container = av.open(fn, mode='w')
        self.stream = self.container.add_stream('mjpeg', rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuvj420p'
        self.stream.bit_rate = 550395
        self.stream.rate = fps

    def write_frame(self, img):
        # Prepare image
        img = np.clip(img, 0, 255)
        img = np.round(img).astype(np.uint8)

        # Convert to VideoFrame, encode and write to container
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        frame = frame.reformat(self.stream.width, self.stream.height, self.stream.pix_fmt)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def write(self, video_frames):
        # Push frames in to video container
        for img in video_frames:
            self.write_frame(img)

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()


if __name__ == '__main__':
    v_reader = VideoReader('scraping/data/prep-final/video/bb5eETSspVI-213.mp4')
    video = v_reader.read(start_time=6., duration=4.0, target_fps=16)
    v_writer = VideoWriter('tmp.mp4', 16, v_reader.width, v_reader.height)
    v_writer.write(video)
