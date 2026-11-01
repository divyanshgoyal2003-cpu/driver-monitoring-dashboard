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


# Where the chin sits on the generic head. The estimator dropped it
# from the solve because a jaw moves it, but the fakes still place it
# so that movement can be simulated.
CHIN_POINT = (0.0, -330.0, -65.0)


def posed_face(yaw=0.0, pitch=0.0, roll=0.0, face=None, model=None,
               distance=3000.0):
    """
    Face whose head pose landmarks are the exact projection of the
    estimator's 3D model at the given angles (HeadPoseEstimator's sign
    convention). RQDecomp3x3 decomposes R = Rz(roll) Ry(yaw) Rx(pitch).

    The chin is placed too, though the estimator no longer solves with
    it, because a real face has one and open_mouth_face needs to be
    able to move it.

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

    solved = estimator.model_points if model is None else np.asarray(model)
    projected = np.vstack([solved, [CHIN_POINT]])

    points, _ = cv2.projectPoints(
        projected,
        rvec,
        tvec,
        estimator.camera_matrix,
        estimator.dist_coeffs
    )

    face = face if face is not None else blank_face()

    indices = tuple(HeadPoseEstimator.LANDMARKS) + (HeadPoseEstimator.CHIN,)

    for index, (x, y) in zip(indices, points.reshape(-1, 2)):
        set_pixel(face, index, x, y)

    place_lips(face, RESTING_MAR)

    return face


# A closed mouth is not a perfectly shut one. Measured over a quiet
# stretch of data/eval/take2.mp4; the peak of a yawn in the same clip
# reaches YAWN_MAR.
RESTING_MAR = 0.02
YAWN_MAR = 0.85


def place_lips(face, mar):
    """
    Put the lip centres between the mouth corners, opened to `mar`.

    The pose estimator reads these to decide whether it is still
    looking at a rigid face, so a posed face needs them even though
    they are not solved against.
    """

    left = face[HeadPoseEstimator.LEFT_MOUTH]
    right = face[HeadPoseEstimator.RIGHT_MOUTH]

    cx = (left.x + right.x) / 2 * WIDTH
    cy = (left.y + right.y) / 2 * HEIGHT
    width = abs(right.x - left.x) * WIDTH

    set_pixel(face, HeadPoseEstimator.MOUTH_TOP, cx, cy - mar * width / 2)
    set_pixel(face, HeadPoseEstimator.MOUTH_BOTTOM, cx, cy + mar * width / 2)


# A head filling the frame, as a camera mounted in a cabin sees it,
# in the millimetres the 3D model is written in. Close enough that
# perspective is doing real work, which is the regime the pose solver
# has to cope with.
DRIVER_DISTANCE = 700.0

# The apparent size of the face in data/eval/take2.mp4, worked back
# from its 131.5 pixels between the eye corners against the model's
# 450mm. A laptop webcam at arm's length, so a far smaller face than
# DRIVER_DISTANCE describes -- and the distance the yawn in
# YAWN_TRAVEL was measured at. Those pixel figures only mean anything
# at this scale: the same landmark travel at DRIVER_DISTANCE is a
# fraction of a face instead of a sizeable part of one.
RECORDED_DISTANCE = 4379.0


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

    nose_p, left_eye, right_eye, left_mouth, right_mouth = (
        estimator.model_points
    )

    def pair(point, scales):
        x, y, z = point
        return [x * scales[0], y * scales[1], z * scales[2]]

    model = np.array([
        list(nose_p),
        pair(left_eye, eye),
        pair(right_eye, eye),
        pair(left_mouth, mouth),
        pair(right_mouth, mouth),
    ])

    return posed_face(yaw=yaw, pitch=pitch, roll=roll, model=model,
                      distance=distance)


# How far each pose landmark travels between a closed mouth and the
# peak of a yawn, in pixels at RECORDED_DISTANCE. Measured off
# data/eval/take2.mp4 rather than invented -- a frontal stretch
# averaged against the peak of the first yawn, in a clip where the
# video shows the head facing straight ahead throughout.
#
# Two things are worth reading off it. Almost nothing is symmetric:
# the left eye corner travels 11.9 pixels while the right travels
# -0.5, and that difference alone is most of the error -- holding the
# left eye still takes the measured yaw from -24.9 back to 0.8. And
# the whole face rises, because a yawn tips the head back.
#
# So this is not really "the jaw moves". It is the landmark mesh
# deforming out of any rigid shape, which leaves solvePnP fitting a
# head that does not exist. Dropping the chin does not cure that; it
# removes the single largest displacement and brings what is left
# inside the thresholds.
YAWN_TRAVEL = {
    HeadPoseEstimator.NOSE: (8.7, -38.8),
    HeadPoseEstimator.CHIN: (9.6, -1.8),
    HeadPoseEstimator.LEFT_EYE: (11.9, -18.3),
    HeadPoseEstimator.RIGHT_EYE: (-0.5, -18.3),
    HeadPoseEstimator.LEFT_MOUTH: (4.9, -20.1),
    HeadPoseEstimator.RIGHT_MOUTH: (13.0, -21.2),
}


def open_mouth_face(yaw=0.0, pitch=0.0, roll=0.0, openness=1.0, seed=0,
                    distance=RECORDED_DISTANCE):
    """
    A face mid-yawn: posed as usual, then deformed by YAWN_TRAVEL.

    solvePnP matches a rigid head to these points and has no way to
    tell a landmark that moved by itself from one that moved because
    the head did, so the deformation comes back as rotation.

    `openness` scales the travel: 1.0 is the full measured yawn, 0.0 a
    closed mouth. `distance` defaults to the one the travel was
    measured at, and changing it without rescaling the travel makes
    the numbers mean something else.
    """

    face = shaped_face(yaw=yaw, pitch=pitch, roll=roll, seed=seed,
                       distance=distance)

    for index, (sideways, down) in YAWN_TRAVEL.items():
        point = face[index]
        set_pixel(face, index,
                  point.x * WIDTH + sideways * openness,
                  point.y * HEIGHT + down * openness)

    place_lips(face, RESTING_MAR + (YAWN_MAR - RESTING_MAR) * openness)

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
