# Advanced Driver Assistance System (ADAS) Project

This project implements a comprehensive ADAS system using multiple sensors (Camera, Radar, and LIDAR) with AI/ML-based object detection and collision avoidance.

## Project Structure

```
adas_project/
├── src/
│   └── adas_system.py      # Main ADAS system implementation
├── utils/
│   └── data_processing.py  # Utility functions for data processing
├── data/                   # Directory for storing sensor data
├── models/                 # Directory for storing trained models
└── docs/                   # Documentation
```

## Features

1. Multi-sensor data integration
   - Camera-based object detection using YOLO
   - Radar data processing
   - LIDAR point cloud processing

2. Sensor Fusion
   - Integration of data from multiple sensors
   - Enhanced object detection accuracy
   - Improved distance and velocity measurements

3. Collision Avoidance
   - Real-time risk assessment
   - Multiple warning levels
   - Automated response system

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLO model:
```bash
# The model will be downloaded automatically when running the code
```

## Usage

Run the main ADAS system:
```bash
python src/adas_system.py
```

## Improvements for Better Efficiency

1. GPU Acceleration
   - Implement GPU support for faster object detection
   - Utilize CUDA for parallel processing

2. Optimized Sensor Fusion
   - Implement Kalman filtering for better sensor fusion
   - Use temporal data for improved tracking

3. Enhanced Risk Assessment
   - Implement machine learning for risk prediction
   - Consider environmental factors in risk assessment

## Data Sources

- Camera data: Custom implementation using OpenCV
- Radar data: Simulated/Real radar sensor data
- LIDAR data: Point cloud data from sensors/simulation

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Ultralytics YOLO
- Other dependencies in requirements.txt 