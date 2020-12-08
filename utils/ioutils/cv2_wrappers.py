import cv2
import numpy as np
from PIL import Image


def opencv_meta(path=None, cap=None):
    if cap is None:
        cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inpt_fps = cap.get(cv2.CAP_PROP_FPS)
    inpt_dur = frame_count / inpt_fps
    meta = {'num_frames': frame_count, 'fps': inpt_fps, 'duration': inpt_dur, 'size': (frame_width, frame_height)}
    if cap is None:
        cap.release()
    return meta


def opencv_loader(path, height=-1, width=-1, fps=-1, start_time=0, duration=-1):
    cap = cv2.VideoCapture(path)
    meta = opencv_meta(cap=cap)

    if duration == -1:
        duration = meta['duration'] - start_time
    else:
        duration = min(duration, meta['duration'] - start_time)

    if fps == -1:
        fps = meta['fps']

    if width == -1:
        width = meta['size'][0]

    if height == -1:
        height = meta['size'][1]

    num_frames = int(duration * fps)
    outp_times = [t for t in np.arange(start_time, start_time + duration, 1. / fps)][:num_frames]
    outp_frames = [int(t * meta['fps']) for t in outp_times]

    buffer = []
    frame_count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count != outp_frames[len(buffer)]:
            continue

        if (meta['size'][1] != height) or (meta['size'][0] != width):
            frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        buffer.append(frame)
        if len(buffer) == len(outp_frames):
            break

    cap.release()
    return buffer
