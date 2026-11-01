import cv2
import numpy as np

from app.config import (
    HEAD_YAW_THRESHOLD,
    HEAD_PITCH_THRESHOLD,
    HEAD_POSE_CALIBRATION_TIME,
    HEAD_POSE_MAX_OFFSET
)


class HeadPoseEstimator:
    """
    Estimates yaw, pitch and roll in degrees.

    Sign convention, from the driver's point of view with an unmirrored
    camera image: negative yaw = turned to the driver's left, positive
    pitch = looking down, positive roll = head tilted clockwise in the
    image.
    """

    # MediaPipe landmark indices
    NOSE = 1
    CHIN = 152
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # The model points below are y-up with +z pointing out of the face,
    # while the camera frame is y-down with +z pointing into the scene.
    # solvePnP therefore returns a 180 degree rotation about x even for a
    # face looking straight at the camera. Decomposing that rotation
    # directly sits on the +-180 degree wrap, where the Euler angles can
    # flip (a frontal face came back as yaw = 180, "RIGHT"). Removing the
    # flip first keeps a frontal face at (0, 0, 0).
    MODEL_TO_CAMERA = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])

    # Six facial landmarks are very nearly coplanar, and a near-planar
    # point set seen through a weak perspective has two poses that
    # reproject almost identically: the real one, and its mirror about
    # the image plane. SOLVEPNP_ITERATIVE is a local optimiser started
    # from a DLT guess, so it falls into whichever basin that guess
    # lands in and gives no warning when it picks the mirror.
    #
    # It does not matter head-on, where the two solutions nearly meet.
    # It matters entirely at the angles this detector exists to catch:
    # measured over a driver holding a steady 63 degree head turn, the
    # iterative solver crossed between the two solutions 31 times in
    # 206 frames, reporting -75 degrees for a head that was at +63.
    # Distraction needs a continuous dwell before it will commit, and
    # every one of those crossings reset the clock -- nine seconds of
    # genuine looking away were reported as 2.5.
    #
    # SQPnP searches for the global optimum instead of descending from
    # a guess, so it does not have a wrong basin to fall into. On the
    # same 206 frames it crossed zero times and its frame-to-frame
    # movement fell from 17.7 degrees to 1.2. It is stateless, so
    # unlike seeding the optimiser from the previous frame it cannot
    # carry one bad estimate forward.
    POSE_SOLVER = cv2.SOLVEPNP_SQPNP

    def __init__(self, frame_width, frame_height,
                 yaw_threshold=HEAD_YAW_THRESHOLD,
                 pitch_threshold=HEAD_PITCH_THRESHOLD,
                 calibration_time=HEAD_POSE_CALIBRATION_TIME,
                 max_offset=HEAD_POSE_MAX_OFFSET):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.calibration_time = calibration_time
        self.max_offset = max_offset

        self.camera_matrix = np.array([
            [frame_width, 0, frame_width / 2],
            [0, frame_width, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

        # Approximate 3D facial model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ], dtype=np.float64)

        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.start_calibration()

    def start_calibration(self):
        """
        Learn a new neutral pose from the next few seconds of samples.
        The driver should look straight ahead while this runs.
        """

        self.calibrated = False
        self.calibration_start = None
        self.calibration_samples = []

    def estimate(self, landmarks, timestamp=None):
        """
        Calibration only runs when a timestamp is passed. Without one the
        raw angles are returned (offsets stay at zero).
        """

        if not landmarks:
            return None

        face = landmarks[0]

        image_points = np.array([
            self._point(face[self.NOSE]),
            self._point(face[self.CHIN]),
            self._point(face[self.LEFT_EYE]),
            self._point(face[self.RIGHT_EYE]),
            self._point(face[self.LEFT_MOUTH]),
            self._point(face[self.RIGHT_MOUTH])
        ], dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=self.POSE_SOLVER
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        pitch, yaw, roll = self._rotation_to_angles(rotation_matrix)

        if timestamp is not None and not self.calibrated:
            self._add_calibration_sample(yaw, pitch, timestamp)

        yaw -= self.yaw_offset
        pitch -= self.pitch_offset

        direction = self._get_direction(yaw, pitch)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "direction": direction,
            "calibrated": self.calibrated
        }

    def _point(self, landmark):

        return (
            landmark.x * self.frame_width,
            landmark.y * self.frame_height
        )

    def _rotation_to_angles(self, rotation_matrix):

        head_rotation = rotation_matrix @ self.MODEL_TO_CAMERA

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(head_rotation)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return pitch, yaw, roll

    def _add_calibration_sample(self, yaw, pitch, timestamp):

        if self.calibration_start is None:
            self.calibration_start = timestamp

        self.calibration_samples.append((yaw, pitch))

        elapsed = timestamp - self.calibration_start

        if elapsed < self.calibration_time:
            return

        if len(self.calibration_samples) < 5:
            return

        samples = np.array(self.calibration_samples)

        self.yaw_offset = float(np.clip(
            np.median(samples[:, 0]), -self.max_offset, self.max_offset
        ))
        self.pitch_offset = float(np.clip(
            np.median(samples[:, 1]), -self.max_offset, self.max_offset
        ))

        self.calibrated = True
        self.calibration_samples = []

    def _get_direction(self, yaw, pitch):

        if yaw < -self.yaw_threshold:
            return "LEFT"

        if yaw > self.yaw_threshold:
            return "RIGHT"

        if pitch < -self.pitch_threshold:
            return "UP"

        if pitch > self.pitch_threshold:
            return "DOWN"

        return "FORWARD"
