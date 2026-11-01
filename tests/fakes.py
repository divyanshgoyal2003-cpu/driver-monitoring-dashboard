"""
Synthetic MediaPipe-style landmarks with known geometry, for unit tests.
Coordinates are given in pixels and stored normalized, like MediaPipe.
"""

import random
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


def posed_face(yaw=0.0, pitch=0.0, roll=0.0, face=None, model=None,
               distance=3000.0):
    """
    Face whose six head pose landmarks are the exact projection of the
    estimator's 3D model at the given angles (HeadPoseEstimator's sign
    convention). RQDecomp3x3 decomposes R = Rz(roll) Ry(yaw) Rx(pitch).

    `model` projects a different 3D face than the one the estimator
    solves with, and `distance` moves it along the optical axis. The
    defaults describe an impossible driver -- the canonical head, far
    enough away to be nearly orthographic -- which every solver gets
    right. Real faces are neither, and the pose ambiguity lives in the
    gap: see shaped_face.
    """

    estimator = HeadPoseEstimator(WIDTH, HEIGHT)

    head_rotation = _rz(roll) @ _ry(yaw) @ _rx(pitch)
    pnp_rotation = head_rotation @ HeadPoseEstimator.MODEL_TO_CAMERA

    rvec, _ = cv2.Rodrigues(pnp_rotation)
    tvec = np.array([0.0, 0.0, distance])

    points, _ = cv2.projectPoints(
        estimator.model_points if model is None else model,
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


# Roughly how far a driver's head sits from a cabin camera, in the
# millimetres the 3D model is written in. Close enough that perspective
# is doing real work, which is the regime the pose solver has to cope
# with.
DRIVER_DISTANCE = 700.0


def shaped_face(yaw=0.0, pitch=0.0, roll=0.0, variation=20.0, seed=0,
                distance=DRIVER_DISTANCE):
    """
    A face at driver distance whose proportions are not the estimator's
    generic model: eye spacing, mouth width, chin length and the depth
    of each feature are scaled by up to `variation` percent, drawn from
    `seed` so the same head comes back on every run.

    The variation is mirrored about the midline, because real heads are
    very nearly symmetric and a lopsided one would read as turned
    whatever the solver did -- that is a face genuinely pointing
    slightly sideways, not a wrong answer.

    This is the gap the pose ambiguity lives in. The solver fits the
    canonical model to landmarks that came from a differently shaped
    head, and the reprojection error that leaves is enough for a
    near-planar point set to fit its own mirror image about as well as
    it fits the truth.
    """

    estimator = HeadPoseEstimator(WIDTH, HEIGHT)
    rng = random.Random(seed)

    def scale():
        return 1.0 + rng.uniform(-variation, variation) / 100.0

    # One scale per feature per axis, shared by the left and right of
    # each pair so the head stays symmetric.
    eye = (scale(), scale(), scale())
    mouth = (scale(), scale(), scale())
    chin = (scale(), scale())

    nose_p, chin_p, left_eye, right_eye, left_mouth, right_mouth = (
        estimator.model_points
    )

    def pair(point, scales):
        x, y, z = point
        return [x * scales[0], y * scales[1], z * scales[2]]

    model = np.array([
        list(nose_p),
        [chin_p[0], chin_p[1] * chin[0], chin_p[2] * chin[1]],
        pair(left_eye, eye),
        pair(right_eye, eye),
        pair(left_mouth, mouth),
        pair(right_mouth, mouth),
    ])

    return posed_face(yaw=yaw, pitch=pitch, roll=roll, model=model,
                      distance=distance)


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


def set_irises(face, ratio, y=300.0):
    """
    Place both eyes' corners and iris centres so each iris sits at
    `ratio` along its corner-to-corner line (0 = image-left corner,
    1 = image-right corner).
    """

    eyes = [
        (468, 33, 133, 530.0),
        (473, 362, 263, 690.0),
    ]

    for iris, left, right, left_x in eyes:
        set_pixel(face, left, left_x, y)
        set_pixel(face, right, left_x + 60.0, y)
        set_pixel(face, iris, left_x + 60.0 * ratio, y)

    return face
