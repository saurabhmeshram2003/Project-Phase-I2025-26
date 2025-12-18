"""
Audio Analysis Module for Quarrel Detection
Detects aggressive audio patterns: screams, shouts, loud arguments
Uses traditional signal processing (no deep learning required)
"""

import numpy as np
import struct
import math
from collections import deque
from scipy.fftpack import fft
from scipy.signal import get_window

# Try to import pyaudio, but make it optional
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("WARNING: PyAudio not available. Audio analysis will be disabled.")
    print("To enable: brew install portaudio && conda run -n quarrel-detection pip install pyaudio")

class AudioAnalyzer:
    """
    Real-time audio analyzer for detecting aggressive sounds.
    Extracts acoustic features without requiring deep learning models.
    """
    
    def __init__(self, 
                 rate=16000,
                 chunk_size=1024,
                 channels=1,
                 energy_threshold=0.02,
                 zcr_threshold=0.15,
                 spectral_threshold=2000):
        """
        Initialize audio analyzer.
        
        Args:
            rate: Sampling rate (Hz)
            chunk_size: Audio buffer size
            channels: Number of audio channels (1=mono)
            energy_threshold: Energy threshold for loud sounds
            zcr_threshold: Zero-crossing rate threshold
            spectral_threshold: Spectral centroid threshold (Hz)
        """
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        
        # Thresholds for aggressive audio detection
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.spectral_threshold = spectral_threshold
        
        # Circular buffer for temporal smoothing
        self.audio_window = deque(maxlen=5)
        
        # PyAudio instance
        self.p = None
        self.stream = None
        
    def start_stream(self):
        """Initialize and start audio stream"""
        if not PYAUDIO_AVAILABLE:
            print("[WARNING] PyAudio not available. Audio analysis disabled.")
            return False
            
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print("[INFO] Audio stream started successfully")
            return True
        except Exception as e:
            print(f"[WARNING] Could not start audio stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop and close audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        print("[INFO] Audio stream stopped")
    
    def read_audio_chunk(self):
        """
        Read one chunk of audio data.
        
        Returns:
            numpy array of audio samples, or None if error
        """
        if not self.stream or not self.stream.is_active():
            return None
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.array(struct.unpack(str(self.chunk_size) + 'h', data))
            # Normalize to [-1, 1]
            audio_data = audio_data / 32768.0
            return audio_data
        except Exception as e:
            print(f"[WARNING] Audio read error: {e}")
            return None
    
    def calculate_energy(self, audio_data):
        """
        Calculate audio energy (loudness).
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Energy value (0-1)
        """
        energy = np.sum(audio_data ** 2) / len(audio_data)
        return energy
    
    def calculate_zero_crossing_rate(self, audio_data):
        """
        Calculate zero-crossing rate (indicates noise/roughness).
        High ZCR indicates abrupt sounds like screams.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            ZCR value (0-1)
        """
        signs = np.sign(audio_data)
        zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(audio_data))
        return zcr
    
    def calculate_spectral_centroid(self, audio_data):
        """
        Calculate spectral centroid (brightness of sound).
        Higher values indicate high-pitched sounds like screams.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Spectral centroid in Hz
        """
        # Apply window to reduce spectral leakage
        windowed = audio_data * get_window('hann', len(audio_data))
        
        # Compute FFT
        spectrum = np.abs(fft(windowed))
        spectrum = spectrum[:len(spectrum)//2]  # Take positive frequencies
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(windowed), 1.0/self.rate)
        freqs = freqs[:len(freqs)//2]
        
        # Calculate centroid
        if np.sum(spectrum) == 0:
            return 0
        
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        return abs(centroid)
    
    def calculate_rms(self, audio_data):
        """
        Calculate Root Mean Square (RMS) - overall amplitude.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            RMS value
        """
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms
    
    def calculate_spectral_rolloff(self, audio_data, rolloff_percent=0.85):
        """
        Calculate spectral rolloff (frequency below which X% of energy is contained).
        
        Args:
            audio_data: Audio samples
            rolloff_percent: Percentage threshold (default 0.85)
            
        Returns:
            Rolloff frequency in Hz
        """
        windowed = audio_data * get_window('hann', len(audio_data))
        spectrum = np.abs(fft(windowed))
        spectrum = spectrum[:len(spectrum)//2]
        
        freqs = np.fft.fftfreq(len(windowed), 1.0/self.rate)
        freqs = freqs[:len(freqs)//2]
        
        total_energy = np.sum(spectrum)
        if total_energy == 0:
            return 0
        
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.where(cumsum >= rolloff_percent * total_energy)[0]
        
        if len(rolloff_idx) == 0:
            return 0
        
        return freqs[rolloff_idx[0]]
    
    def extract_features(self, audio_data):
        """
        Extract all audio features from audio chunk.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Dictionary with feature values
        """
        if audio_data is None or len(audio_data) == 0:
            return None
        
        features = {
            'energy': self.calculate_energy(audio_data),
            'zcr': self.calculate_zero_crossing_rate(audio_data),
            'spectral_centroid': self.calculate_spectral_centroid(audio_data),
            'rms': self.calculate_rms(audio_data),
            'spectral_rolloff': self.calculate_spectral_rolloff(audio_data)
        }
        
        return features
    
    def detect_aggressive_audio(self, features):
        """
        Detect aggressive audio patterns based on features.
        
        Args:
            features: Dictionary with audio features
            
        Returns:
            Tuple: (is_aggressive, confidence_score, reason)
        """
        if features is None:
            return False, 0.0, "No audio data"
        
        aggressive_score = 0.0
        reasons = []
        
        # High energy (loud sounds)
        if features['energy'] > self.energy_threshold:
            aggressive_score += 0.3
            reasons.append("High energy")
        
        # High zero-crossing rate (rough/noisy sounds)
        if features['zcr'] > self.zcr_threshold:
            aggressive_score += 0.25
            reasons.append("High ZCR")
        
        # High spectral centroid (high-pitched screams)
        if features['spectral_centroid'] > self.spectral_threshold:
            aggressive_score += 0.25
            reasons.append("High pitch")
        
        # High RMS (overall loudness)
        if features['rms'] > 0.3:
            aggressive_score += 0.2
            reasons.append("High RMS")
        
        is_aggressive = aggressive_score > 0.5
        reason_str = ", ".join(reasons) if reasons else "Normal"
        
        return is_aggressive, aggressive_score, reason_str
    
    def analyze_real_time(self):
        """
        Analyze current audio chunk in real-time.
        
        Returns:
            Tuple: (is_aggressive, confidence, features, reason)
        """
        audio_data = self.read_audio_chunk()
        if audio_data is None:
            return False, 0.0, None, "No audio"
        
        features = self.extract_features(audio_data)
        is_aggressive, score, reason = self.detect_aggressive_audio(features)
        
        # Temporal smoothing
        self.audio_window.append(score)
        avg_score = np.mean(self.audio_window) if len(self.audio_window) > 0 else 0.0
        
        return is_aggressive, avg_score, features, reason


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("AUDIO ANALYZER - TEST MODE")
    print("="*70)
    
    analyzer = AudioAnalyzer()
    
    if analyzer.start_stream():
        print("\n[INFO] Listening to audio... Press Ctrl+C to stop")
        print("Speak loudly or make noise to test detection\n")
        
        try:
            while True:
                is_aggressive, confidence, features, reason = analyzer.analyze_real_time()
                
                if features:
                    status = "🔴 AGGRESSIVE" if is_aggressive else "🟢 Normal"
                    print(f"{status} | Confidence: {confidence:.2f} | {reason}")
                    print(f"  Energy: {features['energy']:.4f} | ZCR: {features['zcr']:.4f} | "
                          f"Centroid: {features['spectral_centroid']:.0f} Hz")
                    print("-" * 70)
        
        except KeyboardInterrupt:
            print("\n[INFO] Stopping audio analysis...")
        finally:
            analyzer.stop_stream()
    else:
        print("[ERROR] Could not start audio stream")
        print("Make sure you have a microphone connected")
