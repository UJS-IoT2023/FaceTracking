# Face Tracking Project

A high-performance system for real-time face detection and tracking. This repository contains both the Python-based training and the C++ deployment.

## Project Structure
train/: Model training and optimization using Python and uv.

deploy/: High-performance C++ inference 

model/: Storage for exported TorchScript model weights.

data/: Local video files and sample images for testing.

doc/: Documentations

## Technical Specifications
Deep Learning: PyTorch (Training) / LibTorch (Deployment).

Getting Started
Python Training (train/)
Ensure uv is installed on your system.

Navigate to the directory and sync dependencies:

```bash
cd train
uv sync
```

Run the export script to generate the TorchScript model:

```bash
uv run python export.py
```

C++ Deployment (deploy/)
Configure OpenCV and LibTorch paths in CMakeLists.txt.

Use CLion or Visual Studio with the MSVC (x64) toolchain.

Build the project:

```bash
cd deploy
```

## Using CMake to configure and build

```bash
cmake -B cmake-build-debug -G Ninja
cmake --build cmake-build-debug
```

Run the executable to start camera-based tracking.
