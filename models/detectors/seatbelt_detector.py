import math

import cv2
import numpy as np

from app.config import (
    SEATBELT_MIN_ANGLE,
    SEATBELT_MAX_ANGLE,
    SEATBELT_MIN_WIDTH,
    SEATBELT_MAX_WIDTH
)


class SeatbeltDetector:
    """
    Classical computer-vision baseline for seatbelt detection.

    No pretrained seatbelt model ships with the common libraries, and
    training one needs a labelled in-cabin dataset. Until then, this
    looks for the strap's visual signature in the chest region below the
    driver's face: two long, parallel, diagonal edges (the strap's two
    sides) a strap-width apart, crossing the body's centre line. Both
    diagonals are accepted, so left- and right-hand-drive cars both work.

    The centre-line test is what separates a strap from suit lapels and
    shirt collars, which are just as diagonal but converge toward the
    centre and stop there.

    detect() judges a single frame; analytics.seatbelt_monitor pools the
    results over time.
    """

    # Chest region, relative to the face box: from ROI_TOP to ROI_BOTTOM
    # face heights below the chin, and ROI_SIDE face widths either side.
    ROI_SIDE = 0.8
    ROI_TOP = 0.1
    ROI_BOTTOM = 1.6

    # Less chest than this in view (in face heights) and no judgement is
    # made: the strap would not be seen reaching the centre line.
    MIN_VISIBLE_HEIGHT = 0.9

    # An edge must span this fraction of the region's height.
    MIN_EDGE_LENGTH = 0.35

    # The strap's two edges: maximum angle difference (degrees) and
    # minimum overlap along the strap (fraction of the shorter edge).
    PARALLEL_TOLERANCE = 8.0
    MIN_OVERLAP = 0.5

    # The strap (both edges together) must extend at least this share of
    # its horizontal span past the centre line on both sides.
    MIDLINE_SHARE = 0.15

    # Only the longest candidate edges are paired up.
    MAX_EDGES = 40

    def __init__(self, min_angle=SEATBELT_MIN_ANGLE,
                 max_angle=SEATBELT_MAX_ANGLE,
                 min_width=SEATBELT_MIN_WIDTH,
                 max_width=SEATBELT_MAX_WIDTH):

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_width = min_width
        self.max_width = max_width

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame, face_box):
        """
        Returns {"visible", "belt_found", "roi", "strap"}. visible is
        False without a face or with too little of the chest in view.
        strap holds the two edges found, as (x1, y1, x2, y2) in frame
        coordinates.
        """

        result = {
            "visible": False,
            "belt_found": False,
            "roi": None,
            "strap": None
        }

        if face_box is None:
            return result

        roi = self._chest_region(face_box, frame.shape)

        if roi is None:
            return result

        result["visible"] = True
        result["roi"] = roi

        x1, y1, x2, y2 = roi

        # Centre line of the body, taken from the face, in region
        # coordinates.
        midline = (face_box[0] + face_box[2]) / 2 - x1

        edges = self._diagonal_edges(frame[y1:y2, x1:x2])

        strap = self._find_strap(
            edges,
            face_box[2] - face_box[0],
            midline
        )

        if strap is not None:
            result["belt_found"] = True
            result["strap"] = [
                (e["x1"] + x1, e["y1"] + y1, e["x2"] + x1, e["y2"] + y1)
                for e in strap
            ]

        return result

    def _chest_region(self, face_box, shape):

        frame_h, frame_w = shape[:2]
        fx1, fy1, fx2, fy2 = face_box
        face_w, face_h = fx2 - fx1, fy2 - fy1

        if face_w <= 0 or face_h <= 0:
            return None

        x1 = int(max(fx1 - self.ROI_SIDE * face_w, 0))
        x2 = int(min(fx2 + self.ROI_SIDE * face_w, frame_w))
        y1 = int(min(fy2 + self.ROI_TOP * face_h, frame_h))
        y2 = int(min(fy2 + self.ROI_BOTTOM * face_h, frame_h))

        if y2 - y1 < self.MIN_VISIBLE_HEIGHT * face_h:
            return None

        # The strap needs room to be seen crossing the chest.
        if x2 - x1 < 2 * face_w:
            return None

        return x1, y1, x2, y2

    def _diagonal_edges(self, region):

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny thresholds that follow the region's brightness.
        median = float(np.median(gray))
        edges = cv2.Canny(
            gray,
            int(max(0, 0.66 * median)),
            int(min(255, 1.33 * median))
        )

        height = region.shape[0]

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            30,
            minLineLength=int(self.MIN_EDGE_LENGTH * height),
            maxLineGap=int(0.05 * height) + 1
        )

        if lines is None:
            return []

        candidates = []

        for x1, y1, x2, y2 in lines.reshape(-1, 4):

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Fold into (-90, 90]: the direction a segment was found in
            # does not matter.
            if angle <= -90:
                angle += 180
            elif angle > 90:
                angle -= 180

            if not self.min_angle <= abs(angle) <= self.max_angle:
                continue

            candidates.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "angle": angle,
                "length": math.hypot(x2 - x1, y2 - y1)
            })

        candidates.sort(key=lambda e: e["length"], reverse=True)

        return candidates[:self.MAX_EDGES]

    def _find_strap(self, edges, face_width, midline):
        """The longest pair of edges that could be a strap's two sides."""

        best = None
        best_length = 0.0

        for i, a in enumerate(edges):
            for b in edges[i + 1:]:

                if (a["angle"] > 0) != (b["angle"] > 0):
                    continue

                if abs(a["angle"] - b["angle"]) > self.PARALLEL_TOLERANCE:
                    continue

                width = self._distance_to_line(b, a) / face_width

                if not self.min_width <= width <= self.max_width:
                    continue

                if self._overlap(a, b) < self.MIN_OVERLAP * min(
                        a["length"], b["length"]):
                    continue

                if not self._crosses(a, b, midline):
                    continue

                length = min(a["length"], b["length"])

                if length > best_length:
                    best, best_length = (a, b), length

        return best

    def _crosses(self, a, b, midline):
        """Whether the strap with edges a and b crosses the centre line."""

        xs = (a["x1"], a["x2"], b["x1"], b["x2"])
        left, right = min(xs), max(xs)
        span = right - left

        if span <= 0:
            return False

        return min(midline - left, right - midline) >= (
            self.MIDLINE_SHARE * span
        )

    @staticmethod
    def _direction(edge):

        return (
            (edge["x2"] - edge["x1"]) / edge["length"],
            (edge["y2"] - edge["y1"]) / edge["length"]
        )

    def _distance_to_line(self, edge, line):
        """Distance from edge's midpoint to the infinite line through
        `line`."""

        ux, uy = self._direction(line)

        mx = (edge["x1"] + edge["x2"]) / 2 - line["x1"]
        my = (edge["y1"] + edge["y2"]) / 2 - line["y1"]

        return abs(mx * uy - my * ux)

    def _overlap(self, a, b):
        """Length over which b runs alongside a, measured along a."""

        ux, uy = self._direction(a)

        b_start = (b["x1"] - a["x1"]) * ux + (b["y1"] - a["y1"]) * uy
        b_end = (b["x2"] - a["x1"]) * ux + (b["y2"] - a["y1"]) * uy

        low = max(0.0, min(b_start, b_end))
        high = min(a["length"], max(b_start, b_end))

        return high - low
