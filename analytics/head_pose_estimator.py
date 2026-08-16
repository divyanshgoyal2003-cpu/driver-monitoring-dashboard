import cv2
import numpy as np


class HeadPoseEstimator:

    # MediaPipe landmark indices
    NOSE = 1
    CHIN = 152
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def __init__(self, frame_width, frame_height):

        self.frame_width = frame_width
        self.frame_height = frame_height

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

    def estimate(self, landmarks):

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
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles = self._rotation_to_angles(rotation_matrix)

        yaw = angles[1]
        pitch = angles[0]
        roll = angles[2]

        direction = self._get_direction(yaw, pitch)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "direction": direction
        }

    def _point(self, landmark):

        return (
            landmark.x * self.frame_width,
            landmark.y * self.frame_height
        )

    def _rotation_to_angles(self, rotation_matrix):

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # Normalize pitch to the range [-90, 90]
        if pitch > 90:
            pitch -= 180

        if pitch < -90:
            pitch += 180

        return np.array([
            pitch,
            yaw,
            roll
        ])

    def _get_direction(self, yaw, pitch):

        if yaw < -15:
            return "LEFT"

        if yaw > 15:
            return "RIGHT"

        if pitch < -15:
            return "UP"

        if pitch > 15:
            return "DOWN"

        return "FORWARD"