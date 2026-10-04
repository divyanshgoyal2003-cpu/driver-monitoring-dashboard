import time

import cv2

from app.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraManager:
    """
    Handles frame capture from a webcam or a video file.

    Every frame is paired with a timestamp in seconds (last_timestamp).
    For a webcam this is wall-clock time. For a video file it is the
    position within the video, so time-based thresholds stay correct even
    when processing runs slower than the video's frame rate.
    """

    def __init__(self, source=CAMERA_INDEX, width=FRAME_WIDTH,
                 height=FRAME_HEIGHT):

        self.is_file = not isinstance(source, int)
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        if self.is_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.file_fps = fps if fps and fps > 0 else 30.0
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frames_read = 0
        self.last_timestamp = None

    def read_frame(self):

        success, frame = self.cap.read()

        if not success:
            return None

        if self.is_file:
            self.last_timestamp = self.frames_read / self.file_fps
        else:
            self.last_timestamp = time.monotonic()

        self.frames_read += 1

        return frame

    def release(self):
        self.cap.release()
