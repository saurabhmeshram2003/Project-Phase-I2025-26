"""
Configuration file for Quarrel Detection System
Centralized configuration for all modules
"""

import os
from pathlib import Path

# ==== PATHS ====
PROJECT_ROOT = Path(__file__).parent.parent
RAW_VIDEOS_DIR = PROJECT_ROOT / "raw_videos"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR.mkdir(exist_ok=True)

# ==== MODEL PATHS ====
QUARREL_MODEL_PATH = MODEL_DIR / "quarrel_model.h5"

# MobileNet-SSD (Apache 2.0 License - Commercial Friendly)
MOBILENET_PROTOTXT = MODEL_DIR / "mobilenet_ssd" / "deploy.prototxt"
MOBILENET_MODEL = MODEL_DIR / "mobilenet_ssd" / "mobilenet_iter_73000.caffemodel"
PERSON_CLASS_ID = 15  # Person class ID in COCO dataset

# ==== TRAINING CONFIGURATION ====
IMG_SIZE = (224, 224)           # Input image size for CNN
BATCH_SIZE = 32                 # Training batch size
EPOCHS = 20                     # Maximum training epochs
VAL_SPLIT = 0.2                 # Validation split ratio
LEARNING_RATE = 0.001           # Initial learning rate
EARLY_STOP_PATIENCE = 5         # Early stopping patience

# Data augmentation parameters
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.15
HEIGHT_SHIFT_RANGE = 0.15
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.1

# ==== DETECTION CONFIGURATION ====
WINDOW_SIZE = 15                # Frames for temporal smoothing
TEMPORAL_WINDOW = 15            # Alias for WINDOW_SIZE (used in web app)
QUARREL_THRESHOLD = 0.75        # Probability threshold for quarrel detection (increased to reduce false positives)
CAMERA_SOURCE = 0              # 0 for webcam, or video file path
CONFIDENCE_THRESHOLD = 0.3      # Person detection confidence threshold (lower = more sensitive)
IOU_THRESHOLD = 0.4             # YOLO IoU threshold

# Fusion weights for multi-modal detection
CNN_WEIGHT = 0.4                # Weight for CNN prediction (reduced to rely less on appearance)
MOTION_WEIGHT = 0.5             # Weight for motion analysis (increased to focus on behavior)
AUDIO_WEIGHT = 0.1              # Weight for audio analysis

# ==== ALERT CONFIGURATION ====
ENABLE_SOUND_ALERT = True       # Enable/disable sound alert
ENABLE_SNAPSHOT = True          # Save snapshots of quarrel detections
SNAPSHOT_COOLDOWN = 5           # Seconds between snapshots
ALERT_SOUND_PATH = None         # Path to custom alert sound (None = use beep)

# ==== PREPROCESSING CONFIGURATION ====
FRAME_SKIP = 3                  # Extract every Nth frame from videos
MAX_FRAMES_PER_VIDEO = 300      # Maximum frames to extract per video

# ==== DISPLAY CONFIGURATION ====
SHOW_FPS = True                 # Display FPS on video feed
SHOW_CONFIDENCE = True          # Display confidence score
DISPLAY_WIDTH = 1280            # Display window width (None = original)
DISPLAY_HEIGHT = 720            # Display window height (None = original)

# ==== YOLO PERSON CLASS ====
PERSON_CLASS_ID = 0             # COCO dataset person class ID

# ==== MODEL ARCHITECTURE ====
BASE_MODEL = "MobileNetV2"      # Options: MobileNetV2, VGG16, ResNet50
DENSE_UNITS = 128               # Units in dense layer
DROPOUT_RATE = 0.5              # Dropout rate
FREEZE_LAYERS = -20             # Number of layers to freeze from base model

# ==== LOGGING ====
LOG_LEVEL = "INFO"              # Logging level: DEBUG, INFO, WARNING, ERROR
SAVE_TRAINING_PLOT = True       # Save training history plots

# ==== MISCELLANEOUS ====
RANDOM_SEED = 42                # Random seed for reproducibility

def print_config():
    """Print current configuration"""
    print("="*60)
    print("QUARREL DETECTION SYSTEM - CONFIGURATION")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model Path: {QUARREL_MODEL_PATH}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Threshold: {QUARREL_THRESHOLD}")
    print(f"Window Size: {WINDOW_SIZE}")
    print("="*60)

if __name__ == "__main__":
    print_config()
