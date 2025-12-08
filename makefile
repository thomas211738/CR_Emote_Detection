# Name of the virtual environment
VENV = mp_env
PYTHON = python3.10
PIP = $(VENV)/bin/pip
PY = $(VENV)/bin/python

# Default target
all: install

# Create virtual environment
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

# Install dependencies
install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install mediapipe tensorflow opencv-python numpy pillow scikit-learn torch torchvision torchaudio seaborn matplotlib

# Run camera_video.py
run-video:
	$(PY) notebooks/camera_video.py

# Run camera.py
run-photo:
	$(PY) notebooks/camera.py

run-photo-no-tongue:
	$(PY) notebooks/camera_no_tongue.py

# Remove venv
clean:
	rm -rf $(VENV)

.PHONY: all install run-video run-camera clean
