"""
Downloads the trained models the dashboard needs into trained_models/.

trained_models/ is gitignored, so run this once after cloning:

    python download_models.py

Files that are already present are skipped.
"""

import os
import shutil
import urllib.request

from huggingface_hub import hf_hub_download

# (destination, source). A source is either a URL or a
# ("huggingface", repo_id, filename) tuple.
MODELS = [
    (
        "trained_models/face/face_yolov8n.pt",
        ("huggingface", "Bingsu/adetailer", "face_yolov8n.pt")
    ),
    (
        "trained_models/face/face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models"
        "/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    (
        "trained_models/objects/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download"
        "/v8.4.0/yolov8n.pt"
    ),
    (
        "trained_models/emotion/emotion-ferplus-8.onnx",
        "https://github.com/onnx/models/raw/main/validated/vision"
        "/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    ),
]


def download(destination, source):

    if os.path.exists(destination):
        print("Already present:", destination)
        return

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    print("Downloading", destination, "...")

    # Download to a temporary name first so an interrupted download is
    # not mistaken for a finished file on the next run.
    partial = destination + ".part"

    if isinstance(source, tuple):
        _, repo_id, filename = source
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(cached, partial)
    else:
        urllib.request.urlretrieve(source, partial)

    os.replace(partial, destination)

    print("Model saved to:", destination)


if __name__ == "__main__":
    for destination, source in MODELS:
        download(destination, source)
