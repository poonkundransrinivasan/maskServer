# COVID-19 Protocol Monitoring System

A collection of utilities and demos for mask detection and social-distance monitoring.

This repository contains two main modules:

- Mask Detection Module — a socket-based server that receives video frames, runs a face + mask detector, stores results in a local SQLite database, and a small Flask front-end to view detection records and images.
- Social Distance Monitoring Module — a YOLO-based social distancing detector (video input → annotated output / display).

## Table of contents
- About
- Repository layout
- Requirements
- Quick start
  - Mask Detection server + UI
  - Social Distance detector
- Configuration notes and important paths
- Troubleshooting
- Contributing
- License

## About

The project was built as part of a computer vision (BE/Capstone) effort to demonstrate:

- Real-time mask detection using a Keras/TensorFlow model and OpenCV face detector.
- Recording detections (timestamp, address/IP, temperature, reason, saved frame) to SQLite and visualizing via a Flask UI.
- Social distancing detection in video using YOLOv3 and simple centroid-distance checking.

This README documents how the repository is organized, how to configure and run each module, and common issues.

## Repository layout
- `Mask Detection Module/`
  - `Server and UI/Server and UI/`
    - `ServerPyCharm.py` — main socket server and mask-checking logic (entrypoint for mask detection socket service).
    - `frontEnd.py` — Flask app to show entries from `maskDB2.db` using templates in `templates/` and images in `static/`.
    - `maskDB.db` — SQLite databases used by the front-end / server.
    - `MaskDetectionModel/` — trained Keras model directory (expected by `ServerPyCharm.py`).
    - `static/` and `templates/` — web assets and Flask templates used by `frontEnd.py`.
  - `Training and testing/` — Jupyter notebooks used for training/testing the model, example `mask_detector.model` and dataset folder `maskdata/`.

- `Social distance monitoring Module/`
  - `social_distance_detector.py` — script to detect people and flag social distancing violations using a YOLO model.
  - `requirements.txt` — Python packages used by this module.
  - sample video: `pedestrians.mp4`.

Other notebooks and helper files are present across the tree for experimentation and deployment (see the folders above).

## Requirements

The two modules have overlapping but distinct requirements. These are the minimums used during development:

- Python 3.8+ (3.8–3.10 recommended)
- OpenCV (cv2)
- imutils
- numpy
- Flask (for the front-end)
- TensorFlow / Keras (for the mask detection model)
- scipy (used by social distance code)
- sqlite3 (Python standard library)

Social distance module specific requirements (already included in `Social distance monitoring Module/requirements.txt`):

```
imutils==0.5.3
numpy==1.18.5
opencv-python==4.2.0.34
pkg-resources==0.0.0
scipy==1.4.1
```

For the Mask Detection Module you'll need a compatible TensorFlow version (e.g. `tensorflow` or `tensorflow-cpu`) and `tensorflow.keras` available. Exact versions depend on how the model was trained; TensorFlow 2.x is typical.

Note: Some code paths attempt to use GPU acceleration (e.g., YOLO backend target to CUDA). If you plan to use GPU acceleration ensure you have matching OpenCV / CUDA builds and the appropriate NVIDIA drivers + CUDA/cuDNN installed.

## Quick start — recommended workflow (Windows PowerShell)

1) Create a virtual environment and install dependencies

```powershell
cd "c:\Users\vasu2\Desktop\git\maskServer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# install common packages (adjust versions as needed)
pip install flask opencv-python imutils numpy scipy tensorflow
# (optional) install social distance module extras
pip install -r "Social distance monitoring Module\requirements.txt"
```

2) Mask Detection Module — run server and front-end

The mask detection server (`ServerPyCharm.py`) listens on a TCP socket (default port 8090) and expects to receive pickled OpenCV frames from a client. It loads a face detector (Caffe model) and a Keras mask detection model to classify faces.

- Open two terminals. In Terminal A start the socket server:

```powershell
cd "Mask Detection Module\Server and UI\Server and UI"
python ServerPyCharm.py
```

- In Terminal B start the Flask front-end (to view the database entries):

```powershell
cd "Mask Detection Module\Server and UI\Server and UI"
python frontEnd.py
```

Then open a browser to http://localhost:5000/ (or the printed host/port) to view the UI.

Important: `ServerPyCharm.py` currently contains several absolute file paths (for the face detector prototxt, caffemodel, and the Keras model) that must be updated to point to the correct locations on your machine. See the Configuration notes below.

3) Social Distance Monitoring Module

The social distancing script uses YOLOv3 weights and config files. Provide the model files and run like:

```powershell
cd "Social distance monitoring Module"
python social_distance_detector.py --input pedestrians.mp4
```

# or to show camera stream
```powershell
python social_distance_detector.py
```

To write an output video:

```powershell
python social_distance_detector.py --input pedestrians.mp4 --output output.avi
```

The script expects a `MODEL_PATH` configured inside the `TheLazyCoder.social_distancing_config` used by the script. Place `yolov3.cfg`, `yolov3.weights`, and `coco.names` under that folder and update `MODEL_PATH` if needed.

## Configuration notes & important paths

- Mask Detection (ServerPyCharm.py)
  - prototxtPath and weightsPath: these point to the OpenCV DNN face detector (deploy.prototxt / res10_300x300_ssd_iter_140000.caffemodel). In the current file they are absolute Windows paths. Replace them with relative paths or absolute paths that match your environment.
  - maskNet load_model path: currently loads a directory named `MaskDetectionModel`. Ensure the Keras model directory (or `.h5` file) exists and the path is correct.
  - Database: `maskDB2.db` (and others) are located in `Mask Detection Module/Server and UI/Server and UI/` — the Flask app and server open/connect to that DB.
  - Socket port: default `PORT = 8090` — change if needed.

- Social Distance Monitoring
  - `TheLazyCoder/social_distancing_config.py` is referenced by the script and should define `MODEL_PATH` and `MIN_DISTANCE`. Make sure `MODEL_PATH` contains the YOLO files: `yolov3.cfg`, `yolov3.weights`, and `coco.names`.

## Training & testing

Inside `Mask Detection Module/Training and testing/` you'll find notebooks used for training (`MaskDetectionTrainTest.ipynb`) and deployment notebooks. These notebooks show preprocessing, model training, and export steps. If you want to retrain, open the notebooks and run them in a Jupyter environment where TensorFlow is installed.

## Troubleshooting

- Empty frames from camera: ensure camera index is correct (0 or another index) and that no other app is using the camera.
- OpenCV / YOLO GPU issues: OpenCV's DNN CUDA backend requires an OpenCV build compiled with CUDA; pip-built `opencv-python` is usually CPU-only. If you rely on GPU you must install an OpenCV build with CUDA support or use alternative GPU-enabled inference frameworks.
- Absolute paths in code: many paths in `ServerPyCharm.py` are hard-coded. Update these before running or create symlinks / copy files to those absolute locations.
- Virtual environment activation: on Windows PowerShell you might need to change execution policy to allow running Activate.ps1: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (run as Administrator if required).

## Security & privacy

- The server accepts raw pickled frames over TCP. That is convenient for demo clients but can be insecure. Do not expose the socket to untrusted networks and avoid receiving untrusted pickles — unpickling arbitrary data is a code-execution risk. For production, use signed messages, TLS, or a safer serialization format.

## Contributing

If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Open a pull request with a clear description of the change

Small improvements that are helpful:

- Replace absolute paths with relative, configurable settings (config file or environment variables).
- Add a single `requirements.txt` at the repository root.
- Add a small example client that connects to the mask detection socket and sends frames (for local testing).

## License

This repository does not include an explicit license file. Add `LICENSE` to the repo if you want to make the terms explicit. For academic/demo projects, consider MIT or Apache-2.0.

## Contact / Notes

If you want help getting either module running, share the error output and I can help update paths or create helper scripts (for example: a small test client that sends frames to `ServerPyCharm.py` and a consolidated `requirements.txt`).
