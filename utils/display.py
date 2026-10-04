import sys


def screen_size():
    """Primary screen size in pixels, or None when it cannot be read."""

    if sys.platform != "win32":
        return None

    try:
        import ctypes

        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    except Exception:
        return None


def fit_to_screen(width, height, max_fraction=0.9):
    """
    Largest window size with the same aspect ratio that fits on the
    screen. Returns the input size when the screen size is unknown.
    """

    screen = screen_size()

    if not screen:
        return width, height

    screen_w, screen_h = screen

    scale = min(
        1.0,
        screen_w * max_fraction / width,
        screen_h * max_fraction / height
    )

    return int(width * scale), int(height * scale)
