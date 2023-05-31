import cv2
import numpy as np


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        timestamps.append(cur_timestamp)
        cur_timestamp += int(1.0 / fps * 1e3)
        
    frames = np.array(frames)
    timestamps = np.arange(len(frames))
    cur_timestamp = 0

    return frames, np.array(timestamps)