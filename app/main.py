from app.camera import CameraManager
from app.video_processor import VideoProcessor
from app.config import *

import cv2


def main():

    camera = CameraManager()
    processor = VideoProcessor()

    while True:

        frame = camera.read_frame()

        if frame is None:
            break

        processed_frame = processor.process(frame)

        cv2.imshow(WINDOW_NAME, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()