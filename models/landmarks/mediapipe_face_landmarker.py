from types import SimpleNamespace

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeFaceLandmarker:

    # Crop size around a face box for detect_in_region, as a multiple of
    # the box's longer side.
    REGION_SCALE = 2.0

    def __init__(self, model_path, num_faces=1):

        # VIDEO mode tracks the face from frame to frame instead of
        # running full face detection on every frame, which is faster and
        # gives steadier landmarks. It needs a strictly increasing
        # timestamp with every frame.
        self.landmarker = self._create(
            model_path, num_faces, vision.RunningMode.VIDEO
        )

        # Separate IMAGE-mode instance for one-off crops, so they do not
        # disturb the video tracker's timeline.
        self.region_landmarker = self._create(
            model_path, 1, vision.RunningMode.IMAGE
        )

        self.last_timestamp_ms = -1

    @staticmethod
    def _create(model_path, num_faces, running_mode):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        return vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = max(
            int(timestamp * 1000),
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        return result

    def detect_in_region(self, frame, box):
        """
        Find landmarks inside a face box found by another detector.

        MediaPipe's face detector misses faces smaller than roughly a
        third of the frame height, for example when the driver sits far
        from the camera. Cropping around the box makes the face large
        enough to find. Landmarks are returned as a list of faces (like
        result.face_landmarks) in full-frame normalized coordinates.
        """

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = box

        half = max(x2 - x1, y2 - y1) * self.REGION_SCALE / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        left = int(max(cx - half, 0))
        top = int(max(cy - half, 0))
        right = int(min(cx + half, frame_w))
        bottom = int(min(cy + half, frame_h))

        if right - left < 20 or bottom - top < 20:
            return []

        crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=crop
        )

        result = self.region_landmarker.detect(mp_image)

        crop_w, crop_h = right - left, bottom - top
        faces = []

        for face in result.face_landmarks:
            faces.append([
                SimpleNamespace(
                    x=(left + p.x * crop_w) / frame_w,
                    y=(top + p.y * crop_h) / frame_h,
                    z=p.z * crop_w / frame_w
                )
                for p in face
            ])

        return faces

    def close(self):
        self.landmarker.close()
        self.region_landmarker.close()
