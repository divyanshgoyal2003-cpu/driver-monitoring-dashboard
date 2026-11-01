import math

import cv2
import numpy as np

from app.config import (
    HEAD_POSE_MAX_MAR,
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
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # The chin is not among them, and that is the point.
    #
    # solvePnP matches a rigid 3D head to these landmarks, so any
    # landmark that moves relative to the skull is read as the skull
    # having moved. The jaw does exactly that, and the chin is the
    # worst offender: it sits 330mm down the model, the longest lever
    # arm of any point, so a dropped jaw rotates the whole solution.
    #
    # Measured on a recorded yawn, with the driver facing straight
    # ahead the whole time: yaw swung to -28 degrees off its own
    # baseline and stayed there for over two seconds, which is a
    # reported distraction that never happened. Without the chin the
    # same yawn moves yaw by 6.5 degrees, well inside the threshold.
    #
    # Dropping it costs nothing measurable. Pitch is still pinned by
    # the mouth corners below and the eyes above: under a pixel of
    # landmark jitter the recovered pitch error is 0.11 degrees
    # against 0.10 with the chin. Frontal yaw actually improves, from
    # 3.6 +- 2.1 degrees to 1.5 +- 0.7, and five points resist the
    # mirror ambiguity exactly as well as six (0 of 200 either way).
    #
    # The mouth corners move too, but far less, and they earn their
    # place by being the only points below the eyes once the chin has
    # gone.
    CHIN = 152

    # The landmarks actually solved against, in the order model_points
    # lists them.
    LANDMARKS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_MOUTH, RIGHT_MOUTH)

    # The lip centres, used only to notice that the mouth is open. They
    # are not solved against.
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

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
                 max_offset=HEAD_POSE_MAX_OFFSET,
                 max_mar=HEAD_POSE_MAX_MAR):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.calibration_time = calibration_time
        self.max_offset = max_offset
        self.max_mar = max_mar

        self.camera_matrix = np.array([
            [frame_width, 0, frame_width / 2],
            [0, frame_width, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

        # Approximate 3D facial model points, in the same order as
        # LANDMARKS. The chin is deliberately absent: see CHIN.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose
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

        image_points = np.array(
            [self._point(face[index]) for index in self.LANDMARKS],
            dtype=np.float64
        )

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

        reliable = self._mouth_ratio(face) <= self.max_mar

        # A deforming face must not teach the estimator a neutral pose
        # that never existed.
        if timestamp is not None and not self.calibrated and reliable:
            self._add_calibration_sample(yaw, pitch, timestamp)

        yaw -= self.yaw_offset
        pitch -= self.pitch_offset

        direction = self._get_direction(yaw, pitch)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "direction": direction,
            "calibrated": self.calibrated,
            "reliable": reliable
        }

    def _point(self, landmark):

        return (
            landmark.x * self.frame_width,
            landmark.y * self.frame_height
        )

    def _mouth_ratio(self, face):
        """
        How far the mouth is open, as a fraction of its width.

        This estimator fits a rigid head, so an opening mouth is the
        assumption behind it coming apart: the chin and mouth corners
        move on their own, and on a measured yawn even the eye corners
        shifted asymmetrically -- enough on its own to swing the
        reported yaw by 25 degrees with the head facing straight
        ahead. Dropping the chin from the solve reduces that; it
        cannot remove it. So the estimator says when it should not be
        believed instead of pretending otherwise.
        """

        left = self._point(face[self.LEFT_MOUTH])
        right = self._point(face[self.RIGHT_MOUTH])
        top = self._point(face[self.MOUTH_TOP])
        bottom = self._point(face[self.MOUTH_BOTTOM])

        width = math.dist(left, right)

        if width <= 0:
            return 0.0

        return math.dist(top, bottom) / width

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
