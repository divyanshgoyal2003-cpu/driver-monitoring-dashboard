import math

from app.config import GAZE_EYE_RADIUS_RATIO, GAZE_THRESHOLD


class GazeEstimator:
    """
    Eye tracking from MediaPipe's iris landmarks.

    The iris centre's position between the eye corners gives how far the
    eyes are rotated within the head. Adding the head yaw gives where the
    driver is actually looking: a driver whose head is turned 20 degrees
    left but whose eyes are turned 20 degrees right is still looking
    ahead.

    Directions are from the driver's point of view with an unmirrored
    camera image, matching HeadPoseEstimator: negative yaw = left.
    """

    # (iris centre, corner on the image-left side, corner on the
    # image-right side)
    EYES = [
        (468, 33, 133),     # driver's right eye
        (473, 362, 263),    # driver's left eye
    ]

    def __init__(self, eye_radius_ratio=GAZE_EYE_RADIUS_RATIO,
                 threshold=GAZE_THRESHOLD):

        self.eye_radius_ratio = eye_radius_ratio
        self.threshold = threshold

    def estimate(self, landmarks, width, height, head_pose=None):

        if not landmarks:
            return None

        face = landmarks[0]

        # Face models without iris landmarks have only 468 points.
        if len(face) <= max(max(eye) for eye in self.EYES):
            return None

        ratios = [
            self._iris_ratio(face[iris], face[left], face[right],
                             width, height)
            for iris, left, right in self.EYES
        ]

        ratio = sum(ratios) / len(ratios)

        offset = (ratio - 0.5) / self.eye_radius_ratio
        offset = max(-1.0, min(1.0, offset))

        # An iris toward the image's right means the driver is looking to
        # their left, which is negative yaw.
        eye_yaw = -math.degrees(math.asin(offset))

        head_yaw = head_pose["yaw"] if head_pose else 0.0
        gaze_yaw = head_yaw + eye_yaw

        return {
            "iris_ratio": ratio,
            "eye_yaw": eye_yaw,
            "gaze_yaw": gaze_yaw,
            "direction": self._get_direction(gaze_yaw)
        }

    @staticmethod
    def _iris_ratio(iris, left, right, width, height):
        """
        Position of the iris centre along the corner-to-corner line:
        0 at the image-left corner, 1 at the image-right corner.
        """

        lx, ly = left.x * width, left.y * height
        rx, ry = right.x * width, right.y * height
        ix, iy = iris.x * width, iris.y * height

        dx, dy = rx - lx, ry - ly
        length_sq = dx * dx + dy * dy

        if length_sq == 0:
            return 0.5

        return ((ix - lx) * dx + (iy - ly) * dy) / length_sq

    def _get_direction(self, gaze_yaw):

        if gaze_yaw < -self.threshold:
            return "LEFT"

        if gaze_yaw > self.threshold:
            return "RIGHT"

        return "CENTER"
