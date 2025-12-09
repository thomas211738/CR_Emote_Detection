# Clash Royale Emote Detector Project

A Computer Vision project to match user actions to Clash Royale Emotes.

### Note
This model was trained on the two creators (Thomas Yousef, and Joshua Leeds) and therefore works better for them compared to other people. Please note the model may perform worse and not always correctly identify your emote.

## Setup

### 1. Install Dependencies

Create a virtual environment and install all required packages:

```bash
make install
```

This will:
- Create a virtual environment named `mp_env`
- Upgrade pip to the latest version
- Install the following dependencies:
  - mediapipe
  - tensorflow
  - opencv-python
  - numpy
  - pillow
  - scikit-learn
  - torch
  - torchvision
  - torchaudio
  - seaborn
  - matplotlib

## Usage

### Best Model: Video Based

Process video input from camera:

```bash
make run-video
```

This executes `notebooks/camera_video.py`

### Frame-Based Transfer (5 emotes)

Capture photos with camera:

```bash
make run-photo
```

This executes `notebooks/camera.py`

### Frame-Based Transfer (4 emotes) (No Tongue Detection)

Capture photos with camera without tongue detection:

```bash
make run-photo-no-tongue
```

This executes `notebooks/camera_no_tongue.py`

## Cleanup

Remove the virtual environment:

```bash
make clean
```

## Project Structure

```
.
├── Makefile
├── notebooks/
│   ├── camera_video.py
│   ├── camera.py
│   └── camera_no_tongue.py
└── mp_env/  (created after installation)
```



## Authors
This project was developed by Thomas Yousef, and Joshua Leeds
