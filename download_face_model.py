"""
Downloads the trained models required by the dashboard into trained_models/face/.

trained_models/ is gitignored, so run this once after cloning:

    python download_face_model.py
"""

from huggingface_hub import hf_hub_download
import urllib.request
import shutil
import os

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models"
    "/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def download_yolo_face():

    destination = "trained_models/face/face_yolov8n.pt"

    if os.path.exists(destination):
        print("Already present:", destination)
        return

    downloaded_path = hf_hub_download(
        repo_id="Bingsu/adetailer",
        filename="face_yolov8n.pt"
    )

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(downloaded_path, destination)

    print("Model saved to:", destination)


def download_face_landmarker():

    destination = "trained_models/face/face_landmarker.task"

    if os.path.exists(destination):
        print("Already present:", destination)
        return

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    print("Downloading face landmarker...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, destination)

    print("Model saved to:", destination)


if __name__ == "__main__":
    download_yolo_face()
    download_face_landmarker()
