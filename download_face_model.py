from huggingface_hub import hf_hub_download
import shutil
import os

downloaded_path = hf_hub_download(
    repo_id="Bingsu/adetailer",
    filename="face_yolov8n.pt"
)

destination = "trained_models/face/face_yolov8n.pt"

os.makedirs(os.path.dirname(destination), exist_ok=True)
shutil.copy(downloaded_path, destination)

print("Model saved to:", destination)