"""
Shared colours and text helpers for the OpenCV renderers.
Colours are BGR.
"""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

BACKGROUND = (32, 30, 28)
DIVIDER = (70, 66, 62)
TEXT = (235, 235, 235)
MUTED = (150, 146, 140)
GOOD = (110, 200, 80)
WARN = (40, 180, 250)
BAD = (70, 70, 235)
ACCENT = (230, 180, 70)
NEUTRAL_BOX = (190, 190, 190)

LEVEL_COLORS = {
    "SAFE": GOOD,
    "CAUTION": WARN,
    "DANGER": BAD,
}

SEVERITY_COLORS = {
    "critical": BAD,
    "high": BAD,
    "medium": WARN,
    "low": ACCENT,
}


def put_text(image, text, origin, scale=0.5, color=TEXT, thickness=1):

    cv2.putText(
        image,
        text,
        origin,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def text_width(text, scale=0.5, thickness=1):

    (width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)

    return width


def put_text_right(image, text, right_x, y, scale=0.5, color=TEXT,
                   thickness=1):

    x = right_x - text_width(text, scale, thickness)

    put_text(image, text, (x, y), scale, color, thickness)


def put_label(image, text, x, y, color, scale=0.55):
    """
    Text on a filled box in `color`, with its bottom-left corner at
    (x, y). Readable on any background.
    """

    (width, height), baseline = cv2.getTextSize(text, FONT, scale, 1)

    pad = 4
    top = max(y - height - baseline - 2 * pad, 0)
    bottom = top + height + baseline + 2 * pad

    cv2.rectangle(image, (x, top), (x + width + 2 * pad, bottom), color, -1)

    put_text(
        image,
        text,
        (x + pad, bottom - baseline - pad),
        scale,
        (20, 20, 20),
        1
    )


def put_banner(image, text, color, y=0, height=40, scale=0.75):
    """Full-width filled banner with centred text."""

    width = image.shape[1]

    cv2.rectangle(image, (0, y), (width, y + height), color, -1)

    x = (width - text_width(text, scale, 2)) // 2
    baseline = y + (height + int(22 * scale)) // 2

    put_text(image, text, (x, baseline), scale, (255, 255, 255), 2)
