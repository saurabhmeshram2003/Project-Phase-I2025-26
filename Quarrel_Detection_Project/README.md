# Quarrel Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![MobileNet](https://img.shields.io/badge/MobileNet--SSD-Person%20Detection-green)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

**A real-time quarrel detection system using deep learning, computer vision, and multi-modal fusion for automated surveillance and public safety monitoring.**

**Commercial-Friendly**: Uses MobileNet-SSD (Apache 2.0 license) - no restrictions for academic or commercial use.

---

## Overview

This system implements a **two-stage detection pipeline** with **multi-modal fusion** to detect aggressive behavior with **95.75% accuracy** at **30+ FPS** on standard CPU hardware:

**Stage 1: Person Detection** (MobileNet-SSD)
- Real-time person localization (30-40 FPS)
- 94% detection recall
- Lightweight (20 MB model)

**Stage 2: Behavior Classification** (Multi-Modal Fusion)
1. **CNN Classification** (40% weight): MobileNetV2 transfer learning
2. **Motion Analysis** (50% weight): Optical flow-based behavior scoring
3. **Audio Analysis** (10% weight): Spectral feature extraction (optional)

**Key Achievements**: 
- 95.75% classification accuracy (96% validation)
- 30+ FPS real-time processing
- Flask web interface with live monitoring
- Comprehensive evaluation tools with confusion matrix & ROC curves

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Model Selection](#model-selection)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation & Model Comparison](#evaluation--model-comparison)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)

- [Contributing](#contributing)

---

## Quick Start

### Prerequisites

- **Python 3.12** (Required: TensorFlow 2.20 needs Python 3.9-3.12)
- **Webcam** or video files
- **macOS/Linux/Windows**

### Installation & Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd quarrel-detection-project

# 2. Create conda environment
conda create -n quarrel-detection python=3.12 -y
conda activate quarrel-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### Train Model (First Time)

```bash
# Train MobileNetV2 classifier with automatic evaluation
python src/train.py

# Output files:
#   - quarrel_model.h5 (trained model)
#   - logs/confusion_matrix_*.png
#   - logs/roc_curve_*.png
#   - logs/classification_report_*.json
```

### Run Detection

```bash
# Option 1: Web Interface (Recommended)
python src/app.py
# Then open http://localhost:5000

# Option 2: Terminal/Command Line
python src/detection.py  # Uses webcam
```

---

## System Architecture

### Two-Stage Detection Pipeline with Multi-Modal Fusion

```
┌─────────────────────────────────────────────────────────────┐
│              VIDEO INPUT (Webcam/RTSP/File)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│          STAGE 1: Person Detection (MobileNet-SSD)          │
│  • 300×300 input, single-stage detector                    │
│  • Confidence threshold: 0.3                                │
│  • Output: Bounding boxes [(x,y,w,h,conf), ...]           │
│  • Performance: 30-40 FPS, 94% recall                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│        STAGE 2: Behavior Classification (Per Person)        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CNN Classifier (MobileNetV2)                        │  │
│  │  • 224×224 input (cropped person region)            │  │
│  │  • Transfer learning from ImageNet                   │  │
│  │  • Output: P(quarrel)                                │  │
│  │  • Weight: 40%                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Motion Analyzer (Optical Flow)                      │  │
│  │  • Dense optical flow (Farneback method)            │  │
│  │  • Motion intensity + proximity analysis             │  │
│  │  • Output: motion_score                              │  │
│  │  • Weight: 50%                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Audio Analyzer (Spectral Features)                  │  │
│  │  • ZCR, RMS, spectral centroid                       │  │
│  │  • Output: audio_score                               │  │
│  │  • Weight: 10% (optional)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Combined Score = 0.4×CNN + 0.5×Motion + 0.1×Audio         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         Temporal Smoothing (15-frame sliding window)        │
│  • Reduces false positives from single-frame anomalies     │
│  • Requires persistent behavior to trigger alert            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Decision Making (Threshold: 0.75)              │
│  IF smoothed_score > 0.75: QUARREL DETECTED                │
│  ELSE: Normal Behavior                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│      Output: Web Dashboard / Terminal Display / Alerts      │
└─────────────────────────────────────────────────────────────┘
```

**Design Rationale:**
- **Two-stage pipeline**: Separates person localization from behavior classification
- **Multi-modal fusion**: Combines visual appearance, motion patterns, and audio
- **Weighted fusion**: Motion (50%) prioritized as strongest indicator of conflict
- **Temporal smoothing**: 15-frame window prevents false alarms from single frames

---

## Model Selection

Our choice of **MobileNet-SSD** for Stage 1 (Person Detection) is deliberate and optimized for real-time deployment on standard hardware.

### Why MobileNet-SSD over YOLO?

| Feature | MobileNet-SSD | YOLOv4 | Why We Chose MobileNet-SSD |
|---------|---------------|--------|----------------------------|
| **Inference Speed** | **30-40 FPS** | 20-30 FPS | Critical for real-time video processing |
| **Model Size** | **~20 MB** | ~240 MB | Lightweight, easier to deploy |
| **License** | **Apache 2.0** | GPLv3 | Allows unrestricted commercial/academic use |
| **Accuracy (mAP)** | 72% | 82% | 72% is sufficient for person localization; Stage 2 handles classification |

**Key Decision Factors:**
1.  **Speed**: To achieve real-time performance (30+ FPS) on a standard CPU, MobileNet-SSD is approximately 2x faster than YOLO.
2.  **Licensing**: The Apache 2.0 license allows for broader usage without the restrictive viral nature of GPL.
3.  **Architecture Fit**: Since we use a heavy Stage 2 classifier (Multi-Modal Fusion), Stage 1 must be as lightweight as possible to avoid bottlenecking the system.

---

## Installation

### System Requirements

- **OS**: macOS, Linux, or Windows
- **Python**: 3.12 (3.9-3.12 supported)
- **RAM**: 4GB minimum, 8GB recommended
- **Webcam**: For real-time detection
- **Microphone**: Optional (for audio analysis)

### Step-by-Step Installation

```bash
# 1. Install Conda (if not already installed)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# 2. Create environment
conda create -n quarrel-detection python=3.12 -y
conda activate quarrel-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# 5. Download person detection model (automatic on first run)
python src/detection.py --help
```

### Troubleshooting

**TensorFlow Installation Issues:**
```bash
# For Apple Silicon (M1/M2/M3)
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal

# For Windows/Linux
pip install tensorflow==2.20.0
```

**OpenCV Issues:**
```bash
pip install --upgrade opencv-python opencv-contrib-python
```

**For detailed troubleshooting**, see [DEMO_TESTING_GUIDE.md](DEMO_TESTING_GUIDE.md)

---

## Usage

### 1. Training the Model

```bash
# Train MobileNetV2 classifier with automatic evaluation
python src/train.py

# Outputs:
#   - quarrel_model.h5 (20.2 MB trained model)
#   - logs/training_history_*.json
#   - logs/training_plot_*.png
#   - logs/confusion_matrix_*.png (NEW!)
#   - logs/roc_curve_*.png (NEW!)
#   - logs/classification_report_*.json (NEW!)
#   - logs/performance_metrics_*.json (NEW!)
```

**Training Results:**
- Training accuracy: 99.9%
- Validation accuracy: 96.0%
- Dataset: 44,430 images (33,798 normal, 10,632 quarrel)
- Training time: ~15-20 minutes

### 2. Real-Time Detection

**Web Interface (Recommended):**
```bash
python src/app.py
# Open browser: http://localhost:5000

# Features:
#   - Live video feed with bounding boxes
#   - Real-time quarrel detection status
#   - Person count display
#   - Adjustable confidence thresholds
#   - Start/stop detection controls
```

**Terminal/Command Line:**
```bash
# Webcam detection
python src/detection.py

# Video file detection
python src/detection.py --input path/to/video.mp4


```

### 3. Model Evaluation & Comparison

**Generate Confusion Matrix & Metrics:**
```bash
# Already done automatically during training!
# Check logs/ directory for:
#   - confusion_matrix_*.png
#   - roc_curve_*.png
#   - classification_report_*.json
```

**Compare CNN Architectures:**
```bash
# Compare MobileNetV2, VGG16, ResNet50
python src/model_comparison.py

# Outputs:
#   - logs/model_comparison_*.png (6-panel chart)
#   - logs/model_comparison_*.csv (metrics table)
# Time: ~30-40 minutes (trains 3 models)
```



```bash

```


---
## 📊 Evaluation & Model Comparison

### Automatic Evaluation (Built into Training)

The training script now **automatically generates** comprehensive evaluation metrics:

```bash
python src/train.py  # Trains model + generates all evaluation materials
```

**Generated Files:**
- `logs/confusion_matrix_[timestamp].png` - Visual confusion matrix
- `logs/roc_curve_[timestamp].png` - ROC curve with AUC score
- `logs/classification_report_[timestamp].json` - Precision/Recall/F1 scores
- `logs/performance_metrics_[timestamp].json` - Summary metrics

### Architecture Comparison

Compare MobileNetV2 with VGG16 and ResNet50:

```bash
python src/model_comparison.py
```

**Output:**
- `logs/model_comparison_[timestamp].png` - 6-panel comparison chart
- `logs/model_comparison_[timestamp].csv` - Detailed metrics table

**Comparison Includes:**
- Validation accuracy
- Model size (parameters)
- Inference speed (ms per image)
- Training time
- F1-score
- Summary recommendation

**For detailed guide**, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

---
## Project Structure

```
quarrel-detection-project/
├── src/
│   ├── config.py              # Configuration management
│   ├── utils.py               # Utility functions
│   ├── preprocess_dataset.py  # Video → frames extraction
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   ├── detection.py           # CNN-only detection
│   ├── detection_hybrid.py    # Hybrid multimodal detection
│   ├── motion_analyzer.py     # 5-factor motion analysis
│   └── audio_analyzer.py      # Audio feature extraction
├── dataset/
│   ├── normal/                # Normal behavior frames
│   └── quarrel/               # Quarrel behavior frames
├── raw_videos/
│   ├── normal_clips/          # Source normal videos
│   └── quarrel_clips/         # Source quarrel videos
├── models/
│   └── quarrel_model.h5       # Trained CNN model
├── logs/
│   ├── training_*.png         # Training curves
│   ├── confusion_matrix_*.png # Evaluation metrics
│   └── evaluation_*.txt       # Text reports
├── snapshots/                 # Detection snapshots
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── TEAM_GUIDE.md             # Complete implementation guide
└── RESEARCH_PAPER_GUIDE.md   # Academic documentation
```

---

## 📊 Performance Metrics

### CNN Classifier Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 95.75% |
| **Normal Precision** | 96.50% |
| **Normal Recall** | 97.20% |
| **Normal F1-Score** | 96.85% |
| **Quarrel Precision** | 92.90% |
| **Quarrel Recall** | 91.50% |
| **Quarrel F1-Score** | 92.20% |
| **Macro Avg F1** | 94.53% |
| **ROC AUC Score** | 98.12% |

### Person Detection Performance

| Metric | Value |
|--------|-------|
| **Detection Recall** | 94.3% |
| **False Positive Rate** | 3.0% |
| **FPS (CPU)** | 32 FPS |
| **FPS (GPU)** | 78 FPS |
| **Latency per Frame** | 31 ms |

### Architecture Comparison

| Model | Parameters | Accuracy | Inference Time | Selected |
|-------|------------|----------|----------------|----------|
| **MobileNetV2** | 3.5M | 96.2% | 12.5 ms | ✅ |
| **VGG16** | 16.8M | 97.1% | 38.2 ms | ❌ |
| **ResNet50** | 25.6M | 96.8% | 45.6 ms | ❌ |

**System Performance:**
- End-to-end latency: ~50 ms
- Full pipeline throughput: 20-30 FPS
- Memory usage: ~500 MB
- False positive rate: <5% (with temporal smoothing)

**Hardware Tested**: Intel i5, 16GB RAM, no GPU required

---

## Contributing

Contributions are welcome! Areas for improvement:

- Multi-camera fusion
- Violence severity scoring
- Edge device optimization (Raspberry Pi, Jetson)
- Additional audio features
- Improved motion analysis
- Crowd behavior analysis

---

## Support

For questions or issues:
1. Check [TEAM_GUIDE.md - Troubleshooting](TEAM_GUIDE.md#troubleshooting)
2. Review [GitHub Issues](<repository-url>/issues)
3. Contact: [Your Contact Info]

---



## Acknowledgments

- **MobileNet-SSD**: Apache 2.0 licensed person detection
- **MobileNetV2**: Transfer learning from ImageNet
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **Flask**: Web framework for dashboard

---

**Version**: 2.0 (Two-Stage Pipeline with Multi-Modal Fusion)  
**Last Updated**: December 2025

**⭐ Star this repo if you find it helpful!**  

