// Global state
let detectionActive = false;
let updateInterval = null;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const snapshotBtn = document.getElementById('snapshotBtn');
const videoOverlay = document.getElementById('videoOverlay');
const overlayStatus = document.getElementById('overlayStatus');

// Navigation - Bootstrap nav-link
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        // Update active link
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        
        // Show corresponding tab
        const tabName = link.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.style.display = 'none';
        });
        const targetTab = document.getElementById(`${tabName}-tab`);
        if (targetTab) {
            targetTab.style.display = 'block';
        }
    });
});

// Start Detection
startBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/start', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            detectionActive = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            videoOverlay.style.display = 'none';
            
            // Update system status badge with Bootstrap classes
            const statusBadge = document.getElementById('systemStatus');
            statusBadge.innerHTML = '<i class="bi bi-camera-video me-1"></i>Detection Active';
            statusBadge.classList.remove('bg-success');
            statusBadge.classList.add('bg-primary');
            
            // Start updating stats
            startUpdates();
        }
    } catch (error) {
        console.error('Error starting detection:', error);
        showNotification('Failed to start detection', 'danger');
    }
});

// Stop Detection
stopBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            detectionActive = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            videoOverlay.style.display = 'flex';
            
            // Update system status badge
            const statusBadge = document.getElementById('systemStatus');
            statusBadge.innerHTML = '<i class="bi bi-check-circle me-1"></i>System Ready';
            statusBadge.classList.remove('bg-primary');
            statusBadge.classList.add('bg-success');
            
            // Stop updating stats
            stopUpdates();
        }
    } catch (error) {
        console.error('Error stopping detection:', error);
        showNotification('Failed to stop detection', 'danger');
    }
});

// Save Snapshot
snapshotBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/snapshot', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            // Visual feedback
            snapshotBtn.style.transform = 'scale(0.95)';
            setTimeout(() => {
                snapshotBtn.style.transform = 'scale(1)';
            }, 100);
            
            // Show notification
            showNotification('Snapshot saved: ' + data.filename, 'success');
        } else {
            showNotification('Failed to save snapshot: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('Error saving snapshot:', error);
        showNotification('Failed to save snapshot', 'error');
    }
});

// Update Stats
async function updateStats() {
    try {
        // Get detection status
        const statusResponse = await fetch('/api/status');
        const status = await statusResponse.json();
        
        // Get statistics
        const statsResponse = await fetch('/api/stats');
        const stats = await statsResponse.json();
        
        // Update dashboard stats
        document.getElementById('totalDetections').textContent = stats.total_detections;
        document.getElementById('normalCount').textContent = stats.normal_count;
        document.getElementById('quarrelCount').textContent = stats.quarrel_count;
        document.getElementById('fpsValue').textContent = status.fps.toFixed(1);
        
        // Update current status
        const detectionStatusEl = document.getElementById('detectionStatus');
        detectionStatusEl.textContent = status.status;
        detectionStatusEl.className = status.status === 'QUARREL DETECTED' ? 'text-danger' : 'text-success';
        
        document.getElementById('confidenceValue').textContent = 
            (status.confidence * 100).toFixed(0) + '%';
        document.getElementById('personCount').textContent = status.person_count;
        document.getElementById('lastUpdate').textContent = 
            status.timestamp ? new Date(status.timestamp).toLocaleTimeString() : '--:--:--';
        
        // Update score bars
        updateScoreBar('cnn', status.cnn_score);
        updateScoreBar('motion', status.motion_score);
        updateScoreBar('audio', status.audio_score);
        updateScoreBar('combined', status.combined_score);
        
        // Update uptime
        const hours = Math.floor(stats.uptime / 3600);
        const minutes = Math.floor((stats.uptime % 3600) / 60);
        const seconds = stats.uptime % 60;
        document.getElementById('uptime').textContent = 
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Update Score Bar
function updateScoreBar(name, score) {
    const percentage = (score * 100).toFixed(0);
    document.getElementById(`${name}ScoreText`).textContent = percentage + '%';
    const progressBar = document.getElementById(`${name}Progress`);
    progressBar.style.width = percentage + '%';
    
    // Update color for combined score based on threshold
    if (name === 'combined') {
        progressBar.classList.remove('bg-primary', 'bg-danger');
        if (score >= 0.6) {
            progressBar.classList.add('bg-danger');
        } else {
            progressBar.classList.add('bg-primary');
        }
    }
}

// Start Updates
function startUpdates() {
    updateStats(); // Initial update
    updateInterval = setInterval(updateStats, 500); // Update every 500ms
}

// Stop Updates
function stopUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

// Settings - Threshold Slider
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');

thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// Settings - Weight Sliders
const cnnWeightSlider = document.getElementById('cnnWeightSlider');
const cnnWeightValue = document.getElementById('cnnWeightValue');
const motionWeightSlider = document.getElementById('motionWeightSlider');
const motionWeightValue = document.getElementById('motionWeightValue');
const audioWeightSlider = document.getElementById('audioWeightSlider');
const audioWeightValue = document.getElementById('audioWeightValue');

cnnWeightSlider.addEventListener('input', (e) => {
    cnnWeightValue.textContent = parseFloat(e.target.value).toFixed(2);
    normalizeWeights('cnn');
});

motionWeightSlider.addEventListener('input', (e) => {
    motionWeightValue.textContent = parseFloat(e.target.value).toFixed(2);
    normalizeWeights('motion');
});

audioWeightSlider.addEventListener('input', (e) => {
    audioWeightValue.textContent = parseFloat(e.target.value).toFixed(2);
    normalizeWeights('audio');
});

// Normalize weights to sum to 1.0
function normalizeWeights(changedSlider) {
    const cnnWeight = parseFloat(cnnWeightSlider.value);
    const motionWeight = parseFloat(motionWeightSlider.value);
    const audioWeight = parseFloat(audioWeightSlider.value);
    
    const total = cnnWeight + motionWeight + audioWeight;
    
    if (total > 1.0) {
        // Reduce other weights proportionally
        const excess = total - 1.0;
        
        if (changedSlider !== 'cnn') {
            const newCnn = Math.max(0, cnnWeight - (excess * (cnnWeight / (cnnWeight + (changedSlider !== 'motion' ? motionWeight : 0) + (changedSlider !== 'audio' ? audioWeight : 0)))));
            cnnWeightSlider.value = newCnn;
            cnnWeightValue.textContent = newCnn.toFixed(2);
        }
        if (changedSlider !== 'motion') {
            const newMotion = Math.max(0, motionWeight - (excess * (motionWeight / ((changedSlider !== 'cnn' ? cnnWeight : 0) + motionWeight + (changedSlider !== 'audio' ? audioWeight : 0)))));
            motionWeightSlider.value = newMotion;
            motionWeightValue.textContent = newMotion.toFixed(2);
        }
        if (changedSlider !== 'audio') {
            const newAudio = Math.max(0, audioWeight - (excess * (audioWeight / ((changedSlider !== 'cnn' ? cnnWeight : 0) + (changedSlider !== 'motion' ? motionWeight : 0) + audioWeight))));
            audioWeightSlider.value = newAudio;
            audioWeightValue.textContent = newAudio.toFixed(2);
        }
    }
}

// Save Configuration
document.getElementById('saveConfigBtn').addEventListener('click', async () => {
    const config = {
        threshold: parseFloat(thresholdSlider.value),
        cnn_weight: parseFloat(cnnWeightSlider.value),
        motion_weight: parseFloat(motionWeightSlider.value),
        audio_weight: parseFloat(audioWeightSlider.value)
    };
    
    // Validate weights sum to 1.0
    const total = config.cnn_weight + config.motion_weight + config.audio_weight;
    if (Math.abs(total - 1.0) > 0.01) {
        showNotification('Weights must sum to 1.0 (currently: ' + total.toFixed(2) + ')', 'danger');
        return;
    }
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Configuration saved successfully', 'success');
        } else {
            showNotification('Failed to save configuration', 'danger');
        }
    } catch (error) {
        console.error('Error saving config:', error);
        showNotification('Failed to save configuration', 'danger');
    }
});

// Load initial configuration
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        
        thresholdSlider.value = config.threshold;
        thresholdValue.textContent = config.threshold.toFixed(2);
        
        cnnWeightSlider.value = config.cnn_weight;
        cnnWeightValue.textContent = config.cnn_weight.toFixed(2);
        
        motionWeightSlider.value = config.motion_weight;
        motionWeightValue.textContent = config.motion_weight.toFixed(2);
        
        audioWeightSlider.value = config.audio_weight;
        audioWeightValue.textContent = config.audio_weight.toFixed(2);
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

// Show Notification - Using Bootstrap-style toast
function showNotification(message, type = 'info') {
    // Create notification element with Bootstrap classes
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 80px;
        right: 20px;
        z-index: 1050;
        min-width: 300px;
        box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
        animation: slideIn 0.3s ease;
    `;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    
    // Start periodic uptime update
    setInterval(() => {
        if (!detectionActive) {
            updateStats();
        }
    }, 1000);
});

// Add slide animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
