# src/detection.py
"""
Real-time Quarrel Detection System
Integrates YOLO person detection with CNN classification
Features: FPS counter, alerts, snapshot saving, temporal smoothing
"""

import os
import sys
import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from tensorflow.keras.models import load_model

# Import configuration
from config import (
    QUARREL_MODEL_PATH, MOBILENET_PROTOTXT, MOBILENET_MODEL, IMG_SIZE, WINDOW_SIZE,
    QUARREL_THRESHOLD, CAMERA_SOURCE, CONFIDENCE_THRESHOLD,
    ENABLE_SOUND_ALERT, ENABLE_SNAPSHOT, SNAPSHOT_COOLDOWN,
    SNAPSHOTS_DIR, SHOW_FPS, SHOW_CONFIDENCE, PERSON_CLASS_ID,
    DISPLAY_WIDTH, DISPLAY_HEIGHT
)
from utils import play_alert_sound, save_snapshot, calculate_fps
from mobilenet_detector import MobileNetPersonDetector

def preprocess_frame(frame):
    """
    Preprocess frame for CNN input.
    
    Args:
        frame: Input frame (BGR)
        
    Returns:
        Preprocessed frame ready for model prediction
    """
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_norm = frame_resized.astype("float32") / 255.0
    frame_input = np.expand_dims(frame_norm, axis=0)
    return frame_input

def draw_ui(frame, label_text, color, avg_prob, fps=0, box=None):
    """
    Draw UI elements on frame.
    
    Args:
        frame: Input frame
        label_text: Detection label
        color: Text color (BGR)
        avg_prob: Average probability
        fps: Frames per second
        box: Bounding box coordinates (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    
    # Draw semi-transparent overlay at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # Main status text
    cv2.putText(frame, label_text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Probability bar
    if SHOW_CONFIDENCE:
        bar_width = int(300 * avg_prob)
        cv2.rectangle(frame, (20, 55), (320, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 55), (20 + bar_width, 70), color, -1)
        cv2.putText(frame, f"{avg_prob:.2%}", (330, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # FPS counter
    if SHOW_FPS and fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw bounding box around detected people
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    """Main detection loop"""
    print("="*70)
    print("QUARREL DETECTION - REAL-TIME SYSTEM")
    print("="*70)
    print(f"Model: {QUARREL_MODEL_PATH}")
    print(f"YOLO: {YOLO_WEIGHTS}")
    print(f"Camera Source: {CAMERA_SOURCE}")
    print(f"Threshold: {QUARREL_THRESHOLD}")
    print(f"Window Size: {WINDOW_SIZE}")
    print("="*70)
    
    # Check if model exists
    if not Path(QUARREL_MODEL_PATH).exists():
        print(f"\n[ERROR] Model file not found: {QUARREL_MODEL_PATH}")
        print("Please train the model first: python src/train.py")
        sys.exit(1)
    
    # Load trained CNN model
    print("\n[INFO] Loading quarrel detection model...")
    try:
        model = load_model(str(QUARREL_MODEL_PATH))
        print("[SUCCESS] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Load YOLO person detector
    print("[INFO] Loading YOLO model...")
    try:
        yolo_model = YOLO(YOLO_WEIGHTS)
        print("[SUCCESS] YOLO loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        sys.exit(1)

    # Set up video capture
    print(f"[INFO] Opening camera/video source: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera or video source.")
        print("Tips:")
        print("  - For webcam: Use CAMERA_SOURCE = 0 in config.py")
        print("  - For video file: Use CAMERA_SOURCE = 'path/to/video.mp4'")
        sys.exit(1)
    
    # Set display resolution if specified
    if DISPLAY_WIDTH and DISPLAY_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    
    print("[SUCCESS] Camera opened successfully")

    # Sliding window for probabilities
    prob_window = deque(maxlen=WINDOW_SIZE)
    
    # FPS calculation
    frame_times = deque(maxlen=30)
    
    # Alert management
    last_alert_time = 0
    last_snapshot_time = 0
    quarrel_detected = False

    print("\n" + "="*70)
    print("[INFO] Starting real-time detection...")
    print("Press 'q' to quit | Press 's' to save snapshot")
    print("="*70 + "\n")

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or cannot read frame.")
            break

        original_frame = frame.copy()
        
        # Run YOLO on the frame
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Get detections from first result
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                # On COCO, class 0 = "person"
                if cls_id == PERSON_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        if len(boxes) == 0:
            # No person detected
            prob_window.append(0.0)
            avg_prob = np.mean(prob_window) if len(prob_window) > 0 else 0.0
            label_text = "No Person Detected"
            color = (200, 200, 200)
            quarrel_detected = False
            
            draw_ui(frame, label_text, color, avg_prob, 
                   calculate_fps(frame_times))
            
            cv2.imshow("Quarrel Detection System", frame)
            
            frame_times.append(time.time())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Use union of all boxes to capture interaction area
        x1_union = min([b[0] for b in boxes])
        y1_union = min([b[1] for b in boxes])
        x2_union = max([b[2] for b in boxes])
        y2_union = max([b[3] for b in boxes])

        # Clamp to frame size
        h, w, _ = frame.shape
        x1_union = max(0, x1_union)
        y1_union = max(0, y1_union)
        x2_union = min(w - 1, x2_union)
        y2_union = min(h - 1, y2_union)

        person_region = frame[y1_union:y2_union, x1_union:x2_union]
        if person_region.size == 0:
            # Fallback to full frame
            person_region = frame

        # Preprocess and predict
        inp = preprocess_frame(person_region)
        preds = model.predict(inp, verbose=0)[0]  # [p_normal, p_quarrel]
        p_quarrel = float(preds[1])

        prob_window.append(p_quarrel)
        avg_prob = np.mean(prob_window)

        # Determine status
        if avg_prob >= QUARREL_THRESHOLD:
            label_text = "⚠ QUARREL DETECTED"
            color = (0, 0, 255)  # Red
            quarrel_detected = True
            
            # Trigger alert
            current_time = time.time()
            if ENABLE_SOUND_ALERT and (current_time - last_alert_time) > 2:
                play_alert_sound()
                last_alert_time = current_time
            
            # Save snapshot
            if ENABLE_SNAPSHOT and (current_time - last_snapshot_time) > SNAPSHOT_COOLDOWN:
                snapshot_path = save_snapshot(
                    original_frame, "quarrel", avg_prob, SNAPSHOTS_DIR
                )
                print(f"[ALERT] Quarrel detected! Snapshot saved: {snapshot_path}")
                last_snapshot_time = current_time
        else:
            label_text = "✓ Normal Activity"
            color = (0, 255, 0)  # Green
            quarrel_detected = False

        # Draw UI
        draw_ui(frame, label_text, color, avg_prob, 
               calculate_fps(frame_times), 
               box=(x1_union, y1_union, x2_union, y2_union))

        # Display frame
        cv2.imshow("Quarrel Detection System", frame)
        
        # Record frame time
        frame_times.append(time.time())

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quit command received")
            break
        elif key == ord('s'):
            # Manual snapshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = SNAPSHOTS_DIR / f"manual_{timestamp}.jpg"
            cv2.imwrite(str(snapshot_path), original_frame)
            print(f"[INFO] Manual snapshot saved: {snapshot_path}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("[INFO] Detection stopped")
    print("="*70)

if __name__ == "__main__":
    main()
