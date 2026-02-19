// DOM Elements
const anchorInput = document.getElementById('anchorName');
const commandInput = document.getElementById('commandInput');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusBadge = document.getElementById('statusBadge');
const videoOverlay = document.getElementById('videoOverlay');
const videoContainer = document.getElementById('videoContainer');
const fullscreenBtn = document.getElementById('fullscreenBtn');

// Info displays
const anchorDisplay = document.getElementById('anchorDisplay');
const directionDisplay = document.getElementById('directionDisplay');
const faceCountDisplay = document.getElementById('faceCount');
const detectionStatusDisplay = document.getElementById('detectionStatus');

// Quick command chips
const commandChips = document.querySelectorAll('.chip');

// State
let isDetectionActive = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadKnownFaces();
    setTimeout(() => {
        videoOverlay.classList.add('hidden');
    }, 2000);
    
    // Event listeners
    startBtn.addEventListener('click', startDetection);
    stopBtn.addEventListener('click', stopDetection);
    fullscreenBtn.addEventListener('click', toggleFullscreen);
    
    commandChips.forEach(chip => {
        chip.addEventListener('click', () => {
            commandInput.value = chip.dataset.command;
        });
    });
    
    // Enter key handlers
    anchorInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') startDetection();
    });
    
    commandInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') startDetection();
    });
});

// Load known faces count
async function loadKnownFaces() {
    try {
        const response = await fetch('/get_known_faces');
        const data = await response.json();
        
        faceCountDisplay.textContent = `${data.count} faces`;
        
        if (data.count > 0) {
            showNotification('success', 'Database Loaded', `${data.count} registered faces found`);
        }
    } catch (error) {
        console.error('Error loading faces:', error);
        faceCountDisplay.textContent = 'Error';
        showNotification('error', 'Database Error', 'Failed to load face database');
    }
}

// Start detection
async function startDetection() {
    const anchor = anchorInput.value.trim();
    const command = commandInput.value.trim();
    
    if (!command) {
        showNotification('warning', 'Missing Input', 'Please enter a detection command');
        commandInput.focus();
        return;
    }
    
    // Check if it's a direct detection or relative detection
    const hasDirection = command.toLowerCase().includes('right') || 
                        command.toLowerCase().includes('left');
    
    const detectWords = ['detect', 'identify', 'scan', 'find', 'show', 'locate'];
    const isDirectDetection = detectWords.some(word => command.toLowerCase().includes(word)) && !hasDirection;
    
    // For relative detection, anchor is required
    if (hasDirection && !anchor) {
        showNotification('warning', 'Missing Input', 'Please enter your name for left/right detection');
        anchorInput.focus();
        return;
    }
    
    // For direct detection, validate that a person name is mentioned
    if (isDirectDetection) {
        const words = command.split(/\s+/);
        if (words.length < 2) {
            showNotification('warning', 'Invalid Command', 'Please specify a person name (e.g., "detect User1")');
            commandInput.focus();
            return;
        }
    } else if (!hasDirection) {
        showNotification('warning', 'Invalid Command', 'Command must include a person name or "right"/"left" direction');
        commandInput.focus();
        return;
    }
    
    try {
        const response = await fetch('/set_command', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                anchor: anchor,
                command: command
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success' && data.active) {
            isDetectionActive = true;
            
            // Update UI
            startBtn.disabled = true;
            stopBtn.disabled = false;
            commandInput.disabled = true;
            if (hasDirection) {
                anchorInput.disabled = true;
            }
            
            // Update displays
            if (isDirectDetection && data.target_person) {
                anchorDisplay.textContent = 'N/A (Direct Detection)';
                directionDisplay.textContent = 'DIRECT';
                detectionStatusDisplay.textContent = 'Active';
                detectionStatusDisplay.style.color = 'var(--success)';
                showNotification('success', 'Detection Started', `Looking for ${data.target_person}`);
            } else {
                anchorDisplay.textContent = data.anchor;
                directionDisplay.textContent = data.direction ? data.direction.toUpperCase() : 'N/A';
                detectionStatusDisplay.textContent = 'Active';
                detectionStatusDisplay.style.color = 'var(--success)';
                showNotification('success', 'Detection Started', `Looking for person on your ${data.direction}`);
            }
            
            // Update status badge
            const statusDot = statusBadge.querySelector('.status-dot');
            const statusText = statusBadge.querySelector('span');
            statusDot.style.background = 'var(--success)';
            statusText.textContent = 'Detecting';
            
            // Start snapshot polling
            startSnapshotPolling();
        } else {
            showNotification('error', 'Start Failed', 'Could not start detection');
        }
    } catch (error) {
        console.error('Error starting detection:', error);
        showNotification('error', 'Connection Error', 'Failed to communicate with server');
    }
}

// Stop detection
async function stopDetection() {
    try {
        const response = await fetch('/stop_detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'stopped') {
            isDetectionActive = false;
            
            // Update UI
            startBtn.disabled = false;
            stopBtn.disabled = true;
            anchorInput.disabled = false;
            commandInput.disabled = false;
            
            // Update displays
            anchorDisplay.textContent = 'Not set';
            directionDisplay.textContent = 'Not set';
            detectionStatusDisplay.textContent = 'Idle';
            detectionStatusDisplay.style.color = 'var(--text-secondary)';
            
            // Update status badge
            const statusDot = statusBadge.querySelector('.status-dot');
            const statusText = statusBadge.querySelector('span');
            statusDot.style.background = 'var(--warning)';
            statusText.textContent = 'Ready';
            
            showNotification('success', 'Detection Stopped', 'System is now idle');
        }
    } catch (error) {
        console.error('Error stopping detection:', error);
        showNotification('error', 'Stop Failed', 'Failed to stop detection');
    }
}

// Toggle fullscreen
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        videoContainer.requestFullscreen().catch(err => {
            console.error('Fullscreen error:', err);
        });
        videoContainer.classList.add('fullscreen');
    } else {
        document.exitFullscreen();
        videoContainer.classList.remove('fullscreen');
    }
}

// Listen for fullscreen changes
document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
        videoContainer.classList.remove('fullscreen');
    }
});

// Notification system
function showNotification(type, title, message) {
    const container = document.getElementById('notificationContainer');
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    const icons = {
        success: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                  </svg>`,
        error: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="15" y1="9" x2="9" y2="15"/>
                  <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>`,
        warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                   <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                   <line x1="12" y1="9" x2="12" y2="13"/>
                   <line x1="12" y1="17" x2="12.01" y2="17"/>
                 </svg>`
    };
    
    notification.innerHTML = `
        <div class="notification-icon">
            ${icons[type] || icons.success}
        </div>
        <div class="notification-content">
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        </div>
    `;
    
    container.appendChild(notification);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 4000);
}

// Handle video feed errors
document.getElementById('videoFeed').addEventListener('error', () => {
    showNotification('error', 'Video Feed Error', 'Failed to load video stream');
});

// Poll for snapshots
let snapshotCheckInterval = null;

function startSnapshotPolling() {
    snapshotCheckInterval = setInterval(checkForSnapshot, 2000);
}

function stopSnapshotPolling() {
    if (snapshotCheckInterval) {
        clearInterval(snapshotCheckInterval);
        snapshotCheckInterval = null;
    }
}

async function checkForSnapshot() {
    try {
        const response = await fetch('/get_snapshot');
        const data = await response.json();
        
        if (data.has_snapshot) {
            displaySnapshot(data);
            stopSnapshotPolling(); // Stop polling once snapshot is shown
        }
    } catch (error) {
        console.error('Error checking for snapshot:', error);
    }
}

function displaySnapshot(data) {
    const validationDiv = document.getElementById('snapshotValidation');
    const snapshotImage = document.getElementById('snapshotImage');
    const snapshotPerson = document.getElementById('snapshotPerson');
    const snapshotPosition = document.getElementById('snapshotPosition');
    
    snapshotImage.src = 'data:image/jpeg;base64,' + data.image;
    snapshotPerson.textContent = data.person_name;
    snapshotPosition.textContent = data.position_info;
    
    validationDiv.style.display = 'block';
    validationDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Close snapshot
document.getElementById('closeSnapshot').addEventListener('click', () => {
    document.getElementById('snapshotValidation').style.display = 'none';
    discardSnapshot();
});

// Confirm snapshot
document.getElementById('confirmSnapshot').addEventListener('click', async () => {
    const personName = document.getElementById('snapshotPerson').textContent;
    
    try {
        const response = await fetch('/save_snapshot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                confirmed: true,
                person_name: personName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('success', 'Snapshot Saved', data.message);
            document.getElementById('snapshotValidation').style.display = 'none';
            
            // Ask about sketch generation
            if (data.ask_sketch) {
                showSketchModal(data.filename);
            } else {
                startSnapshotPolling(); // Resume polling
            }
        } else {
            showNotification('error', 'Save Failed', data.message);
        }
    } catch (error) {
        showNotification('error', 'Error', 'Failed to save snapshot');
    }
});

// Discard snapshot
document.getElementById('discardSnapshot').addEventListener('click', () => {
    discardSnapshot();
    document.getElementById('snapshotValidation').style.display = 'none';
    startSnapshotPolling(); // Resume polling
});

async function discardSnapshot() {
    try {
        await fetch('/save_snapshot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirmed: false })
        });
    } catch (error) {
        console.error('Error discarding snapshot:', error);
    }
}

// Sketch generation modal
let currentImagePath = '';

function showSketchModal(imagePath) {
    currentImagePath = imagePath;
    document.getElementById('sketchModal').style.display = 'flex';
}

document.getElementById('generateSketchBtn').addEventListener('click', async () => {
    const progressDiv = document.getElementById('sketchProgress');
    const resultDiv = document.getElementById('sketchResult');
    const generateBtn = document.getElementById('generateSketchBtn');
    const skipBtn = document.getElementById('skipSketchBtn');
    
    generateBtn.disabled = true;
    skipBtn.disabled = true;
    progressDiv.style.display = 'block';
    resultDiv.style.display = 'none';
    
    try {
        const response = await fetch('/generate_sketch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: currentImagePath })
        });
        
        const data = await response.json();
        
        progressDiv.style.display = 'none';
        
        if (data.success) {
            resultDiv.innerHTML = `
                <div class="success-message">
                    ✓ Sketch generated successfully!
                </div>
                <div class="sketch-paths">
                    <div class="sketch-path">Edge Sketch: ${data.sketch_path}</div>
                    <div class="sketch-path">Pencil Sketch: ${data.pencil_sketch_path}</div>
                </div>
            `;
            resultDiv.style.display = 'block';
            showNotification('success', 'Sketch Generated', 'Schematic diagrams created successfully');
            
            setTimeout(() => {
                document.getElementById('sketchModal').style.display = 'none';
                resetSketchModal();
                startSnapshotPolling(); // Resume polling
            }, 4000);
        } else {
            resultDiv.innerHTML = `
                <div class="error-message" style="background: rgba(255, 51, 102, 0.1); border: 1px solid rgba(255, 51, 102, 0.3); border-radius: 10px; padding: 1rem; color: var(--danger); text-align: center;">
                    ✗ ${data.message}
                </div>
            `;
            resultDiv.style.display = 'block';
            showNotification('error', 'Generation Failed', data.message);
        }
    } catch (error) {
        progressDiv.style.display = 'none';
        showNotification('error', 'Error', 'Failed to generate sketch');
    } finally {
        generateBtn.disabled = false;
        skipBtn.disabled = false;
    }
});

document.getElementById('skipSketchBtn').addEventListener('click', () => {
    document.getElementById('sketchModal').style.display = 'none';
    resetSketchModal();
    startSnapshotPolling(); // Resume polling
});

function resetSketchModal() {
    document.getElementById('sketchProgress').style.display = 'none';
    document.getElementById('sketchResult').style.display = 'none';
    document.getElementById('sketchResult').innerHTML = '';
    currentImagePath = '';
}

// Start polling when page loads and detection is active
window.addEventListener('load', () => {
    if (isDetectionActive) {
        startSnapshotPolling();
    }
});

// Also start polling when detection starts
const originalStartDetection = startDetection;
async function startDetection() {
    await originalStartDetection();
    if (isDetectionActive) {
        startSnapshotPolling();
    }
}

// Stop polling when detection stops
const originalStopDetection = stopDetection;
async function stopDetection() {
    stopSnapshotPolling();
    await originalStopDetection();
}
