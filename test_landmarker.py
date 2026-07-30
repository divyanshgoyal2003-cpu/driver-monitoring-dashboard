import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "trained_models/face/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)

print("✅ Face Landmarker Loaded Successfully!")