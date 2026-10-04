"""
Synthetic MediaPipe-style landmarks with known geometry, for unit tests.
Coordinates are given in pixels and stored normalized, like MediaPipe.
"""

from types import SimpleNamespace

import cv2
import numpy as np

from analytics.blink_detector import BlinkDetector
from analytics.head_pose_estimator import HeadPoseEstimator
from analytics.yawn_detector import YawnDetector

WIDTH = 1280
HEIGHT = 720
NUM_LANDMARKS = 478


def blank_face():
    return [SimpleNamespace(x=0.5, y=0.5, z=0.0)
            for _ in range(NUM_LANDMARKS)]


def set_pixel(face, index, x, y):
    face[index] = SimpleNamespace(x=x / WIDTH, y=y / HEIGHT, z=0.0)


def set_eye(face, indices, cx, cy, ear, eye_width=60.0):
    """
    Place the six EAR landmarks, in BlinkDetector order, so the eye has
    exactly the requested aspect ratio.
    """

    p1, p2, p3, p4, p5, p6 = indices

    half_w = eye_width / 2
    # EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|) = 2 * half_h / eye_width
    half_h = ear * eye_width / 2

    set_pixel(face, p1, cx - half_w, cy)
    set_pixel(face, p4, cx + half_w, cy)
    set_pixel(face, p2, cx - 10, cy - half_h)
    set_pixel(face, p6, cx - 10, cy + half_h)
    set_pixel(face, p3, cx + 10, cy - half_h)
    set_pixel(face, p5, cx + 10, cy + half_h)


def set_mouth(face, cx, cy, mar, mouth_width=100.0):

    set_pixel(face, YawnDetector.MOUTH_LEFT, cx - mouth_width / 2, cy)
    set_pixel(face, YawnDetector.MOUTH_RIGHT, cx + mouth_width / 2, cy)
    set_pixel(face, YawnDetector.MOUTH_TOP, cx, cy - mar * mouth_width / 2)
    set_pixel(face, YawnDetector.MOUTH_BOTTOM, cx, cy + mar * mouth_width / 2)


def make_face(ear=0.30, mar=0.05):

    face = blank_face()

    set_eye(face, BlinkDetector.LEFT_EYE, 560, 300, ear)
    set_eye(face, BlinkDetector.RIGHT_EYE, 720, 300, ear)
    set_mouth(face, 640, 450, mar)

    return face


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a), np.cos(a)]])


def _ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)],
                     [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]])


def posed_face(yaw=0.0, pitch=0.0, roll=0.0, face=None):
    """
    Face whose six head pose landmarks are the exact projection of the
    estimator's 3D model at the given angles (HeadPoseEstimator's sign
    convention). RQDecomp3x3 decomposes R = Rz(roll) Ry(yaw) Rx(pitch).
    """

    estimator = HeadPoseEstimator(WIDTH, HEIGHT)

    head_rotation = _rz(roll) @ _ry(yaw) @ _rx(pitch)
    pnp_rotation = head_rotation @ HeadPoseEstimator.MODEL_TO_CAMERA

    rvec, _ = cv2.Rodrigues(pnp_rotation)
    tvec = np.array([0.0, 0.0, 3000.0])

    points, _ = cv2.projectPoints(
        estimator.model_points,
        rvec,
        tvec,
        estimator.camera_matrix,
        estimator.dist_coeffs
    )

    face = face if face is not None else blank_face()

    indices = [
        HeadPoseEstimator.NOSE,
        HeadPoseEstimator.CHIN,
        HeadPoseEstimator.LEFT_EYE,
        HeadPoseEstimator.RIGHT_EYE,
        HeadPoseEstimator.LEFT_MOUTH,
        HeadPoseEstimator.RIGHT_MOUTH
    ]

    for index, (x, y) in zip(indices, points.reshape(-1, 2)):
        set_pixel(face, index, x, y)

    return face


def feed(detector, face, start, duration, fps=30.0):
    """
    Call detector.update with the same face for `duration` seconds.
    Returns (last result, next timestamp).
    """

    result = None
    t = start
    steps = int(round(duration * fps))

    for _ in range(steps):
        landmarks = [face] if face is not None else []
        result = detector.update(landmarks, WIDTH, HEIGHT, t)
        t += 1.0 / fps

    return result, t
