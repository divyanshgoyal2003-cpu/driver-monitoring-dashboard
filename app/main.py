import argparse

import cv2

from app.camera import CameraManager
from app.config import CAMERA_INDEX, WINDOW_NAME
from app.video_processor import VideoProcessor
from utils.alert_sound import AlertSound
from utils.display import fit_to_screen


def parse_args():

    parser = argparse.ArgumentParser(
        description="Unified Driver Monitoring Dashboard"
    )

    parser.add_argument(
        "--source",
        default=str(CAMERA_INDEX),
        help="webcam index (e.g. 0) or path to a video file"
    )

    parser.add_argument(
        "--mute",
        action="store_true",
        help="start with alert sounds off (press S to toggle)"
    )

    return parser.parse_args()


def parse_source(source):
    return int(source) if source.isdigit() else source


def main():

    args = parse_args()

    camera = CameraManager(parse_source(args.source))
    processor = VideoProcessor()
    alert_sound = AlertSound()

    if args.mute:
        alert_sound.enabled = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_sized = False

    try:
        while True:

            frame = camera.read_frame()

            if frame is None:
                break

            output = processor.process(frame, camera.last_timestamp)

            if not window_sized:
                cv2.resizeWindow(
                    WINDOW_NAME,
                    *fit_to_screen(output.shape[1], output.shape[0])
                )
                window_sized = True

            cv2.imshow(WINDOW_NAME, output)

            alert_sound.update(processor.state["risk"]["alerts"])

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            if key == ord('l'):
                processor.show_landmarks = not processor.show_landmarks

            if key == ord('c'):
                processor.recalibrate()

            if key == ord('s'):
                alert_sound.toggle()

            # Closing the window with its X button ends the session.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        processor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
