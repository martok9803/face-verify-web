# Face Verify Web

Upload two images (ID + Photo) and verify if they belong to the same person.
Uses dlib (HOG) for detection and DeepFace (VGG-Face) for comparison.

## Setup (Ubuntu)
```bash
sudo apt update
sudo apt install -y python3-venv build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
