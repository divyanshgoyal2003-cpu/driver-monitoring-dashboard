import argparse
import sys

import cv2

from app.camera import CameraManager
from app.config import (
    CAMERA_INDEX,
    WINDOW_NAME,
    DATABASE_PATH,
    WEATHER_MULTIPLIERS,
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE
)
from app.video_processor import VideoProcessor
from database.database import Database
from database.session_logger import SessionLogger
from risk_engine.context import ContextProvider
from utils.alert_sound import AlertSound
from utils.display import fit_to_screen
from utils.weather import WeatherService


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
        choices=sorted(WEATHER_MULTIPLIERS),
        help="declare the driving conditions, which raise what a lapse "
             "costs. Overrules the forecast for the whole session; "
             "press W in the window to change it or hand control back"
    )

    parser.add_argument(
        "--latitude",
        type=float,
        default=WEATHER_LATITUDE,
        help="where to fetch the weather for. Without this (and "
             "--longitude) no forecast is fetched and nothing leaves "
             "the machine"
    )

    parser.add_argument(
        "--longitude",
        type=float,
        default=WEATHER_LONGITUDE,
        help="see --latitude"
    )

    parser.add_argument(
        "--no-forecast",
        action="store_true",
        help="never fetch the weather, even with coordinates set"
    )

    parser.add_argument(
        "--speed",
        type=float,
        help="vehicle speed in km/h. Not sensed: needs an OBD-II link"
    )

    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="do not lift dim frames before detection"
    )

    return parser.parse_args()


def parse_source(source):
    return int(source) if source.isdigit() else source


def main():

    args = parse_args()

    try:
        camera = CameraManager(parse_source(args.source))

    except RuntimeError as error:
        # A mistyped filename or a webcam another program already has
        # open. Both are ordinary mistakes and deserve a line rather
        # than a traceback, the same as the evaluation harness gives.
        print(error, file=sys.stderr)
        return 1

    weather_service = WeatherService(
        latitude=None if args.no_forecast else args.latitude,
        longitude=None if args.no_forecast else args.longitude
    )

    if weather_service.start():
        print(f"Fetching the weather for "
              f"{args.latitude:.3f}, {args.longitude:.3f}. "
              f"Press W to overrule it.")

    context_provider = ContextProvider(
        weather=args.weather,
        speed=args.speed,
        hour=args.hour,
        weather_service=weather_service
    )

    processor = VideoProcessor(
        dynamic_weights=not args.static_weights,
        context_provider=context_provider,
        enhance=not args.no_enhance
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

            # The forecast answers for a point, not for this road. The
            # driver is the one who can see out of the window.
            if key == ord('w'):
                declared = context_provider.cycle_weather()

                print(
                    "Weather: back to the forecast"
                    if declared is None
                    else f"Weather: {declared.replace('_', ' ')} "
                         f"(driver override)"
                )

            # Closing the window with its X button ends the session.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        processor.close()
        weather_service.stop()
        cv2.destroyAllWindows()

        if logger:
            logger.close()
            database.close()
            print(f"Session {logger.session_id} saved to {args.db}. "
                  f"View it with: streamlit run dashboard/dashboard.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
