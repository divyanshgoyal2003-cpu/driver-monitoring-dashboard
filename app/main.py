import argparse

import cv2

from app.camera import CameraManager
from app.config import CAMERA_INDEX, WINDOW_NAME, DATABASE_PATH
from app.video_processor import VideoProcessor
from database.database import Database
from database.session_logger import SessionLogger
from risk_engine.context import ContextProvider
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

    parser.add_argument(
        "--db",
        default=DATABASE_PATH,
        help=f"session database for the analytics dashboard "
             f"(default: {DATABASE_PATH})"
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="do not save this session to the database"
    )

    parser.add_argument(
        "--static-weights",
        action="store_true",
        help="score with the flat penalty table instead of weighing "
             "each risk by severity, detector confidence and context"
    )

    parser.add_argument(
        "--hour",
        type=int,
        choices=range(24),
        metavar="0-23",
        help="pretend it is this hour, to demonstrate the circadian "
             "weighting without waiting for 3am"
    )

    parser.add_argument(
        "--weather",
        default="clear",
        help="driving conditions. Not sensed: an in-cabin camera cannot "
             "see the road, so this is a manual override"
    )

    parser.add_argument(
        "--speed",
        type=float,
        help="vehicle speed in km/h. Not sensed: needs an OBD-II link"
    )

    return parser.parse_args()


def parse_source(source):
    return int(source) if source.isdigit() else source


def main():

    args = parse_args()

    camera = CameraManager(parse_source(args.source))

    processor = VideoProcessor(
        dynamic_weights=not args.static_weights,
        context_provider=ContextProvider(
            weather=args.weather,
            speed=args.speed,
            hour=args.hour
        )
    )

    alert_sound = AlertSound()

    if args.mute:
        alert_sound.enabled = False

    database = None
    logger = None

    if not args.no_log:
        database = Database(args.db)
        logger = SessionLogger(database, args.source)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_sized = False

    try:
        while True:

            frame = camera.read_frame()

            if frame is None:
                break

            output = processor.process(frame, camera.last_timestamp)

            if logger:
                logger.log(processor.state)

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

        if logger:
            logger.close()
            database.close()
            print(f"Session {logger.session_id} saved to {args.db}. "
                  f"View it with: streamlit run dashboard/dashboard.py")


if __name__ == "__main__":
    main()
