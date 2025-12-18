"""
Utility functions for Quarrel Detection System
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras History object or dict with 'accuracy', 'loss', etc.
        save_path: Path to save the plot (if None, displays plot)
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Train Accuracy', marker='o')
    ax1.plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history['loss'], label='Train Loss', marker='o')
    ax2.plot(history['val_loss'], label='Val Loss', marker='s')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Training plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (2D array)
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_training_history(history, save_path):
    """
    Save training history to JSON file.
    
    Args:
        history: Keras History object or dict
        save_path: Path to save JSON file
    """
    if hasattr(history, 'history'):
        history = history.history
    
    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            history_serializable[key] = value.tolist()
        elif isinstance(value, list):
            history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
        else:
            history_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(history_serializable, f, indent=4)
    
    print(f"[INFO] Training history saved to: {save_path}")

def load_training_history(load_path):
    """
    Load training history from JSON file.
    
    Args:
        load_path: Path to JSON file
        
    Returns:
        Dictionary with training history
    """
    with open(load_path, 'r') as f:
        history = json.load(f)
    return history

def create_timestamp():
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_metrics(metrics_dict):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics_dict: Dictionary with metric names and values
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print("="*60)

def save_snapshot(frame, label, confidence, output_dir):
    """
    Save a snapshot of detected quarrel.
    
    Args:
        frame: Image frame (numpy array)
        label: Detection label
        confidence: Confidence score
        output_dir: Directory to save snapshot
    """
    import cv2
    timestamp = create_timestamp()
    filename = f"quarrel_{timestamp}_conf{confidence:.2f}.jpg"
    filepath = Path(output_dir) / filename
    
    cv2.imwrite(str(filepath), frame)
    return filepath

def calculate_fps(frame_times, window_size=30):
    """
    Calculate FPS from frame timestamps.
    
    Args:
        frame_times: List of frame timestamps
        window_size: Number of frames to average
        
    Returns:
        Average FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    recent_times = frame_times[-window_size:]
    if len(recent_times) < 2:
        return 0.0
    
    time_diff = recent_times[-1] - recent_times[0]
    if time_diff == 0:
        return 0.0
    
    fps = (len(recent_times) - 1) / time_diff
    return fps

def play_alert_sound():
    """
    Play alert sound (cross-platform).
    Uses system beep or plays custom sound file.
    """
    import platform
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            os.system("afplay /System/Library/Sounds/Glass.aiff &")
        elif system == "Linux":
            os.system("paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga &")
        elif system == "Windows":
            import winsound
            winsound.Beep(1000, 500)  # 1000 Hz for 500ms
    except Exception as e:
        # Fallback to terminal beep
        print("\a")  # ASCII bell character

if __name__ == "__main__":
    # Test plot functions
    print("Testing utility functions...")
    
    # Test training history plot
    dummy_history = {
        'accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.65, 0.75, 0.8, 0.85, 0.87],
        'loss': [0.6, 0.4, 0.3, 0.2, 0.15],
        'val_loss': [0.65, 0.45, 0.35, 0.25, 0.2]
    }
    
    print("✓ Utility functions ready")
