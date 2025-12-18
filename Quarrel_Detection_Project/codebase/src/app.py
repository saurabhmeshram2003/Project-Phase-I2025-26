from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
from collections import deque
import json
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from motion_analyzer import MotionAnalyzer
from audio_analyzer import AudioAnalyzer
from mobilenet_detector import MobileNetPersonDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quarrel-detection-secret-key'

# Global variables
camera = None
detection_active = False
current_frame = None
detection_results = {
    'status': 'Normal',
    'confidence': 0.0,
    'cnn_score': 0.0,
    'motion_score': 0.0,
    'audio_score': 0.0,
    'combined_score': 0.0,
    'fps': 0.0,
    'person_count': 0,
    'timestamp': '',
    'detections': []
}
frame_lock = threading.Lock()
results_lock = threading.Lock()

# Models
cnn_model = None
person_detector = None  # MobileNet-SSD detector
motion_analyzer = None
audio_analyzer = None

# Temporal smoothing
score_history = deque(maxlen=TEMPORAL_WINDOW)
stats = {
    'total_detections': 0,
    'quarrel_count': 0,
    'normal_count': 0,
    'uptime': 0,
    'start_time': time.time()
}


def load_models():
    """Load all ML models"""
    global cnn_model, person_detector, motion_analyzer, audio_analyzer
    
    try:
        # Load CNN model
        if os.path.exists(QUARREL_MODEL_PATH):
            cnn_model = tf.keras.models.load_model(str(QUARREL_MODEL_PATH))
            print(f"✓ CNN model loaded from {QUARREL_MODEL_PATH}")
        else:
            print(f"⚠ CNN model not found at {QUARREL_MODEL_PATH}")
            cnn_model = None
        
        # Load MobileNet-SSD person detector (Apache 2.0 - Commercial Friendly)
        person_detector = MobileNetPersonDetector(
            str(MOBILENET_PROTOTXT),
            str(MOBILENET_MODEL),
            CONFIDENCE_THRESHOLD
        )
        print(f"✓ MobileNet-SSD loaded (Apache 2.0 License - Commercial Use OK)")
        
        # Initialize motion analyzer
        motion_analyzer = MotionAnalyzer()
        print("✓ Motion analyzer initialized")
        
        # Initialize audio analyzer
        try:
            audio_analyzer = AudioAnalyzer()
            print("✓ Audio analyzer initialized")
        except Exception as e:
            print(f"⚠ Audio analyzer failed: {e}")
            audio_analyzer = None
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


def process_frame(frame):
    """Process a single frame through the detection pipeline"""
    global detection_results, score_history
    
    if frame is None or cnn_model is None or person_detector is None:
        return frame
    
    start_time = time.time()
    results_dict = {
        'status': 'Normal',
        'confidence': 0.0,
        'cnn_score': 0.0,
        'motion_score': 0.0,
        'audio_score': 0.0,
        'combined_score': 0.0,
        'person_count': 0,
        'detections': []
    }
    
    try:
        # MobileNet-SSD person detection
        detection_results_mobilenet = person_detector(frame)
        detected_boxes = detection_results_mobilenet[0]
        
        # Debug: Print detection info
        num_detections = len(detected_boxes.xyxy)
        if num_detections > 0:
            print(f"[DEBUG] Detected {num_detections} person(s)")
        
        boxes = []
        boxes_with_conf = []
        cnn_scores = []
        
        if len(detected_boxes.xyxy) > 0:
            for i, box in enumerate(detected_boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                conf = float(detected_boxes.conf[i])
                
                boxes.append((x1, y1, x2, y2))  # For motion analyzer
                boxes_with_conf.append((x1, y1, x2, y2, conf))  # For drawing
                
                # Extract and classify person crop
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    resized = cv2.resize(person_crop, IMG_SIZE)
                    normalized = resized / 255.0
                    input_data = np.expand_dims(normalized, axis=0)
                    
                    cnn_pred = cnn_model.predict(input_data, verbose=0)[0][0]
                    cnn_scores.append(float(cnn_pred))
        
        # CNN score (max across all persons)
        cnn_score = max(cnn_scores) if cnn_scores else 0.0
        results_dict['cnn_score'] = cnn_score
        results_dict['person_count'] = len(boxes)
        
        # Motion analysis
        motion_score = 0.0
        if motion_analyzer:
            motion_result = motion_analyzer.calculate_motion_score(frame, boxes)
            motion_score = motion_result['total_score'] if isinstance(motion_result, dict) else motion_result
        results_dict['motion_score'] = motion_score
        
        # Audio analysis
        audio_score = 0.0
        if audio_analyzer:
            try:
                _, audio_score, _, _ = audio_analyzer.analyze_real_time()
            except:
                audio_score = 0.0
        results_dict['audio_score'] = audio_score
        
        # Weighted fusion
        combined_score = (CNN_WEIGHT * cnn_score + 
                         MOTION_WEIGHT * motion_score + 
                         AUDIO_WEIGHT * audio_score)
        
        # Temporal smoothing
        score_history.append(combined_score)
        smoothed_score = np.mean(score_history)
        
        results_dict['combined_score'] = float(smoothed_score)
        results_dict['confidence'] = float(smoothed_score)
        
        # Determine status
        if smoothed_score >= QUARREL_THRESHOLD:
            results_dict['status'] = 'QUARREL DETECTED'
            stats['quarrel_count'] += 1
        else:
            results_dict['status'] = 'Normal'
            stats['normal_count'] += 1
        
        stats['total_detections'] += 1
        
        # Draw on frame
        for (x1, y1, x2, y2, conf) in boxes_with_conf:
            color = (0, 0, 255) if results_dict['status'] == 'QUARREL DETECTED' else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence label
            label = f"{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            results_dict['detections'].append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf
            })
        
        # Add status overlay
        status_color = (0, 0, 255) if results_dict['status'] == 'QUARREL DETECTED' else (0, 255, 0)
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Status: {results_dict['status']}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Confidence: {smoothed_score:.2f}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"CNN: {cnn_score:.2f} | Motion: {motion_score:.2f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Audio: {audio_score:.2f} | Persons: {len(boxes)}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        results_dict['fps'] = float(fps)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Confidence bar
        bar_width = int(smoothed_score * 350)
        bar_color = (0, 0, 255) if smoothed_score >= QUARREL_THRESHOLD else (0, 255, 0)
        cv2.rectangle(frame, (420, 40), (420 + bar_width, 70), bar_color, -1)
        cv2.rectangle(frame, (420, 40), (770, 70), (255, 255, 255), 2)
        
        results_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update global results
        with results_lock:
            detection_results.update(results_dict)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    return frame


def camera_stream():
    """Background thread for camera capture and processing"""
    global camera, current_frame, detection_active
    
    camera = cv2.VideoCapture(CAMERA_SOURCE)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        if not detection_active:
            time.sleep(0.1)
            continue
            
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            time.sleep(0.1)
            continue
        
        # Process frame
        processed_frame = process_frame(frame.copy())
        
        # Update current frame
        with frame_lock:
            current_frame = processed_frame.copy()


def generate_frames():
    """Generator for video streaming"""
    global current_frame
    
    while True:
        if current_frame is None:
            time.sleep(0.1)
            continue
        
        with frame_lock:
            frame = current_frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection"""
    global detection_active
    detection_active = True
    return jsonify({'success': True, 'message': 'Detection started'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_active
    detection_active = False
    return jsonify({'success': True, 'message': 'Detection stopped'})


@app.route('/api/status')
def get_status():
    """Get current detection status"""
    with results_lock:
        return jsonify(detection_results)


@app.route('/api/stats')
def get_stats():
    """Get detection statistics"""
    stats['uptime'] = int(time.time() - stats['start_time'])
    return jsonify(stats)


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Get or update configuration"""
    global QUARREL_THRESHOLD, CNN_WEIGHT, MOTION_WEIGHT, AUDIO_WEIGHT
    
    if request.method == 'POST':
        data = request.json
        if 'threshold' in data:
            QUARREL_THRESHOLD = float(data['threshold'])
        if 'cnn_weight' in data:
            CNN_WEIGHT = float(data['cnn_weight'])
        if 'motion_weight' in data:
            MOTION_WEIGHT = float(data['motion_weight'])
        if 'audio_weight' in data:
            AUDIO_WEIGHT = float(data['audio_weight'])
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    
    return jsonify({
        'threshold': QUARREL_THRESHOLD,
        'cnn_weight': CNN_WEIGHT,
        'motion_weight': MOTION_WEIGHT,
        'audio_weight': AUDIO_WEIGHT
    })


@app.route('/api/snapshot', methods=['POST'])
def save_snapshot():
    """Save current frame as snapshot"""
    global current_frame
    
    if current_frame is None:
        return jsonify({'success': False, 'message': 'No frame available'})
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(str(SNAPSHOTS_DIR), filename)
        
        with frame_lock:
            cv2.imwrite(filepath, current_frame)
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    print("="*60)
    print("Quarrel Detection System - Web Interface")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    load_models()
    
    # Start camera thread
    print("\nStarting camera stream...")
    camera_thread = threading.Thread(target=camera_stream, daemon=True)
    camera_thread.start()
    
    print("\nSystem ready!")
    print(f"Open http://localhost:5000 in your browser")
    print("="*60)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
