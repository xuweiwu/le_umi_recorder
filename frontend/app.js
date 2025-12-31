/**
 * WebXR Controller Tracking Application
 * Streams Meta Quest controller poses to backend server via WebSocket
 */

class ControllerTracker {
    constructor() {
        // WebXR state
        this.xrSession = null;
        this.xrRefSpace = null;
        this.gl = null;

        // WebSocket connection
        this.ws = null;
        this.connected = false;

        // Configuration
        // Auto-detect protocol (ws:// for HTTP, wss:// for HTTPS)
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const defaultUrl = `${wsProtocol}//${window.location.hostname}:${window.location.port || 8000}/ws`;

        this.config = {
            trackLeft: true,
            trackRight: true,
            coordinateSystem: 'local',
            targetRate: 60,
            serverUrl: defaultUrl
        };

        // Performance tracking
        this.frameCount = 0;
        this.lastFrameTime = performance.now();
        this.actualRate = 0;

        // Controller data cache
        this.lastPoseData = null;

        // WebXR support flags
        this.arSupported = false;
        this.vrSupported = false;

        // Throttling for updates
        this.lastSendTime = 0;
        this.sendInterval = 1000 / this.config.targetRate;

        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        // Set default server URL in the input field
        document.getElementById('server-url').value = this.config.serverUrl;

        this.setupUI();
        this.checkWebXRSupport();
    }

    /**
     * Compile WebGL shader
     */
    compileShader(gl, source, type) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    /**
     * Initialize WebGL for rendering
     */
    initWebGL() {
        const gl = this.gl;

        // Vertex shader - transforms positions
        const vertexShaderSource = `
            attribute vec3 aPosition;
            uniform mat4 uProjectionMatrix;
            uniform mat4 uViewMatrix;
            uniform mat4 uModelMatrix;

            void main() {
                gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aPosition, 1.0);
            }
        `;

        // Fragment shader - solid colors
        const fragmentShaderSource = `
            precision mediump float;
            uniform vec4 uColor;

            void main() {
                gl_FragColor = uColor;
            }
        `;

        // Compile shaders
        const vertexShader = this.compileShader(gl, vertexShaderSource, gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER);

        if (!vertexShader || !fragmentShader) {
            console.error('Failed to compile shaders');
            return false;
        }

        // Create program
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
            return false;
        }

        // Store program and locations
        this.shaderProgram = program;
        this.shaderLocations = {
            position: gl.getAttribLocation(program, 'aPosition'),
            projectionMatrix: gl.getUniformLocation(program, 'uProjectionMatrix'),
            viewMatrix: gl.getUniformLocation(program, 'uViewMatrix'),
            modelMatrix: gl.getUniformLocation(program, 'uModelMatrix'),
            color: gl.getUniformLocation(program, 'uColor')
        };

        // Create coordinate axes geometry (lines for X, Y, Z axes)
        this.createAxesGeometry();

        // Enable depth test
        gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0.0, 0.0, 0.0, 0.0); // Transparent black

        console.log('WebGL initialized successfully');
        return true;
    }

    /**
     * Create coordinate axes geometry
     */
    createAxesGeometry() {
        const gl = this.gl;

        // Axes: X (red), Y (green), Z (blue)
        // Each axis is a line from origin to direction
        const axisLength = 0.1; // 10cm axes

        const vertices = new Float32Array([
            // X axis (red)
            0, 0, 0,  axisLength, 0, 0,
            // Y axis (green)
            0, 0, 0,  0, axisLength, 0,
            // Z axis (blue)
            0, 0, 0,  0, 0, axisLength
        ]);

        this.axesBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.axesBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

        this.axesVertexCount = 6; // 3 axes * 2 points each
    }

    /**
     * Setup UI event listeners
     */
    setupUI() {
        // Connect button
        document.getElementById('connect-btn').addEventListener('click', () => {
            this.config.serverUrl = document.getElementById('server-url').value;
            this.connectWebSocket();
        });

        // AR button (passthrough)
        document.getElementById('ar-btn').addEventListener('click', () => {
            this.startXR('immersive-ar');
        });

        // VR button
        document.getElementById('vr-btn').addEventListener('click', () => {
            this.startXR('immersive-vr');
        });

        // Disconnect button
        document.getElementById('disconnect-btn').addEventListener('click', () => {
            this.disconnect();
        });

        // Configuration updates
        document.getElementById('track-left').addEventListener('change', (e) => {
            this.config.trackLeft = e.target.checked;
        });

        document.getElementById('track-right').addEventListener('change', (e) => {
            this.config.trackRight = e.target.checked;
        });

        document.querySelectorAll('input[name="coord-system"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.config.coordinateSystem = e.target.value;
            });
        });

        document.getElementById('update-rate').addEventListener('change', (e) => {
            this.config.targetRate = parseInt(e.target.value);
            this.sendInterval = 1000 / this.config.targetRate;
        });
    }

    /**
     * Check if WebXR is supported
     */
    async checkWebXRSupport() {
        if ('xr' in navigator) {
            try {
                const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
                const arSupported = await navigator.xr.isSessionSupported('immersive-ar');

                this.vrSupported = vrSupported;
                this.arSupported = arSupported;

                if (arSupported) {
                    this.updateStatus('WebXR AR supported - Passthrough available', 'info');
                } else if (vrSupported) {
                    this.updateStatus('WebXR VR supported - Manual passthrough only', 'info');
                } else {
                    this.updateStatus('WebXR not supported on this device', 'error');
                }
            } catch (err) {
                this.updateStatus('Error checking WebXR support: ' + err.message, 'error');
            }
        } else {
            this.updateStatus('WebXR not available - Use Meta Quest Browser', 'error');
        }
    }

    /**
     * Connect to WebSocket server
     */
    connectWebSocket() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.updateStatus('Already connected', 'warning');
            return;
        }

        this.updateStatus('Connecting to server...', 'info');
        this.updateWSStatus('Connecting...', 'connecting');

        try {
            this.ws = new WebSocket(this.config.serverUrl);

            this.ws.onopen = () => {
                this.connected = true;
                this.updateStatus('Connected to server', 'success');
                this.updateWSStatus('Connected', 'connected');
                document.getElementById('ar-btn').disabled = !this.arSupported;
                document.getElementById('vr-btn').disabled = !this.vrSupported;
                document.getElementById('connect-btn').disabled = true;
                document.getElementById('disconnect-btn').disabled = false;

                // Send initial handshake
                this.sendMessage({
                    type: 'handshake',
                    client: 'quest3',
                    timestamp: Date.now() / 1000
                });
            };

            this.ws.onclose = () => {
                this.connected = false;
                this.updateStatus('Disconnected from server', 'error');
                this.updateWSStatus('Disconnected', 'disconnected');
                document.getElementById('ar-btn').disabled = true;
                document.getElementById('vr-btn').disabled = true;
                document.getElementById('connect-btn').disabled = false;
                document.getElementById('disconnect-btn').disabled = true;
            };

            this.ws.onerror = (error) => {
                this.updateStatus('WebSocket error: ' + error, 'error');
                this.updateWSStatus('Error', 'disconnected');
            };

            this.ws.onmessage = (event) => {
                this.handleServerMessage(event.data);
            };

        } catch (err) {
            this.updateStatus('Failed to connect: ' + err.message, 'error');
            this.updateWSStatus('Failed', 'disconnected');
        }
    }

    /**
     * Handle messages from server
     */
    handleServerMessage(data) {
        try {
            const message = JSON.parse(data);
            console.log('Server message:', message);

            if (message.type === 'ack') {
                console.log('Server acknowledged connection');
            }
        } catch (err) {
            console.error('Failed to parse server message:', err);
        }
    }

    /**
     * Send message via WebSocket
     */
    sendMessage(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    /**
     * Start WebXR session
     * @param {string} mode - 'immersive-ar' for passthrough or 'immersive-vr' for VR
     */
    async startXR(mode = 'immersive-ar') {
        if (!navigator.xr) {
            this.updateStatus('WebXR not available', 'error');
            return;
        }

        try {
            // Create WebGL context
            const canvas = document.createElement('canvas');
            this.gl = canvas.getContext('webgl', { xrCompatible: true });

            // Request XR session
            const sessionInit = {
                requiredFeatures: ['local-floor'],
                optionalFeatures: mode === 'immersive-ar' ? ['hand-tracking'] : []
            };

            this.xrSession = await navigator.xr.requestSession(mode, sessionInit);
            this.currentMode = mode;

            const modeLabel = mode === 'immersive-ar' ? 'AR (Passthrough)' : 'VR';
            this.updateStatus(`${modeLabel} session started`, 'success');
            document.getElementById('vr-btn').disabled = true;
            document.getElementById('ar-btn').disabled = true;

            // Setup WebGL
            await this.gl.makeXRCompatible();

            // Initialize WebGL shaders and geometry
            if (!this.initWebGL()) {
                throw new Error('Failed to initialize WebGL');
            }

            // Create XR WebGL layer with alpha for transparency
            const xrGlLayer = new XRWebGLLayer(this.xrSession, this.gl, {
                alpha: true  // Enable alpha channel
            });

            // Update render state - use baseLayer approach
            this.xrSession.updateRenderState({
                baseLayer: xrGlLayer
            });

            // Check if passthrough is available via environment blend mode
            if (this.xrSession.environmentBlendMode === 'alpha-blend' ||
                this.xrSession.environmentBlendMode === 'additive') {
                console.log('✓ Passthrough enabled via blend mode:', this.xrSession.environmentBlendMode);
            } else {
                console.log('Environment blend mode:', this.xrSession.environmentBlendMode);
                console.log('Note: Passthrough may not be available. Use Quest settings to enable passthrough in browser.')
            }

            // Get reference space
            //this.xrRefSpace = await this.xrSession.requestReferenceSpace('local-floor');
            // Get the base reference space
            const baseRefSpace = await this.xrSession.requestReferenceSpace('local-floor');
            // Create a -90° rotation around X-axis to convert Y-up → Z-up
            const sqrt2_2 = Math.SQRT1_2;
            const rotation = new DOMPointReadOnly(-sqrt2_2, 0, 0, sqrt2_2); // -90° around X
            const transform = new XRRigidTransform({ x: 0, y: 0, z: 0 }, rotation);

            // Create the Z-up reference space
            this.xrRefSpace = baseRefSpace.getOffsetReferenceSpace(transform);

            // Start render loop
            this.xrSession.requestAnimationFrame(this.onXRFrame.bind(this));

            // Handle session end
            this.xrSession.addEventListener('end', () => {
                this.xrSession = null;
                this.currentMode = null;
                this.updateStatus('XR session ended', 'info');
                document.getElementById('vr-btn').disabled = !this.vrSupported || !this.connected;
                document.getElementById('ar-btn').disabled = !this.arSupported || !this.connected;
            });

        } catch (err) {
            this.updateStatus('Failed to start VR: ' + err.message, 'error');
            console.error('XR Error:', err);
        }
    }

    /**
     * Render coordinate axes at a given transform
     */
    renderAxes(viewMatrix, projectionMatrix, transform) {
        const gl = this.gl;

        // Use shader program
        gl.useProgram(this.shaderProgram);

        // Set matrices
        gl.uniformMatrix4fv(this.shaderLocations.projectionMatrix, false, projectionMatrix);
        gl.uniformMatrix4fv(this.shaderLocations.viewMatrix, false, viewMatrix);
        gl.uniformMatrix4fv(this.shaderLocations.modelMatrix, false, transform.matrix);

        // Bind axes buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, this.axesBuffer);
        gl.enableVertexAttribArray(this.shaderLocations.position);
        gl.vertexAttribPointer(this.shaderLocations.position, 3, gl.FLOAT, false, 0, 0);

        // Draw X axis (red)
        gl.uniform4f(this.shaderLocations.color, 1.0, 0.0, 0.0, 1.0);
        gl.drawArrays(gl.LINES, 0, 2);

        // Draw Y axis (green)
        gl.uniform4f(this.shaderLocations.color, 0.0, 1.0, 0.0, 1.0);
        gl.drawArrays(gl.LINES, 2, 2);

        // Draw Z axis (blue)
        gl.uniform4f(this.shaderLocations.color, 0.0, 0.0, 1.0, 1.0);
        gl.drawArrays(gl.LINES, 4, 2);
    }

    /**
     * Render the VR scene
     */
    renderScene(frame) {
        const session = frame.session;
        const gl = this.gl;
        const pose = frame.getViewerPose(this.xrRefSpace);

        if (!pose) return;

        const layer = session.renderState.baseLayer;
        gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);

        // Clear with transparent background (for passthrough)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Render for each view (eye)
        for (const view of pose.views) {
            const viewport = layer.getViewport(view);
            gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);

            const projectionMatrix = view.projectionMatrix;
            const viewMatrix = view.transform.inverse.matrix;

            // Render controller axes
            const inputSources = session.inputSources;
            for (const source of inputSources) {
                if (source.targetRayMode !== 'tracked-pointer') continue;

                const gripPose = frame.getPose(source.gripSpace, this.xrRefSpace);
                if (gripPose) {
                    this.renderAxes(viewMatrix, projectionMatrix, gripPose.transform);
                }
            }
        }
    }

    /**
     * WebXR frame callback - runs at display refresh rate
     */
    onXRFrame(time, frame) {
        const session = frame.session;
        session.requestAnimationFrame(this.onXRFrame.bind(this));

        // Render the scene
        this.renderScene(frame);

        // Update frame rate
        this.updateFrameRate();

        // Throttle sending based on target rate
        const now = performance.now();
        if (now - this.lastSendTime < this.sendInterval) {
            return;
        }
        this.lastSendTime = now;

        // Get controller poses
        const poseData = this.getControllerPoses(frame);

        if (poseData && this.connected) {
            this.sendMessage(poseData);
            this.updateDebug(poseData);
        }
    }

    /**
     * Extract controller pose data from XR frame
     */
    getControllerPoses(frame) {
        const inputSources = frame.session.inputSources;
        const timestamp = Date.now() / 1000;

        const controllers = {};

        for (const source of inputSources) {
            if (source.targetRayMode !== 'tracked-pointer') continue;

            const handedness = source.handedness; // 'left' or 'right'

            // Check if we should track this controller
            if (handedness === 'left' && !this.config.trackLeft) continue;
            if (handedness === 'right' && !this.config.trackRight) continue;

            // Get grip pose (controller position/orientation)
            const gripPose = frame.getPose(source.gripSpace, this.xrRefSpace);
            console.log({gripPose});
            if (gripPose) {
                const pos = gripPose.transform.position;
                const ori = gripPose.transform.orientation;

                controllers[handedness] = {
                    position: [pos.x, pos.y, pos.z],
                    orientation: [ori.x, ori.y, ori.z, ori.w], // quaternion
                    buttons: this.getButtonStates(source.gamepad)
                };
            }
        }

        // Only send if we have controller data
        if (Object.keys(controllers).length === 0) {
            return null;
        }

        return {
            type: 'pose',
            timestamp: timestamp,
            coordinate_system: this.config.coordinateSystem,
            controllers: controllers
        };
    }

    /**
     * Get button states from gamepad
     */
    getButtonStates(gamepad) {
        if (!gamepad) return {};

        const buttons = {};
        gamepad.buttons.forEach((button, index) => {
            buttons[index] = {
                pressed: button.pressed,
                touched: button.touched,
                value: button.value
            };
        });

        return buttons;
    }

    /**
     * Update frame rate display
     */
    updateFrameRate() {
        this.frameCount++;
        const now = performance.now();
        const elapsed = now - this.lastFrameTime;

        if (elapsed >= 1000) {
            this.actualRate = Math.round((this.frameCount * 1000) / elapsed);
            document.getElementById('frame-rate').textContent = this.actualRate;

            this.frameCount = 0;
            this.lastFrameTime = now;
        }
    }

    /**
     * Update debug display
     */
    updateDebug(data) {
        this.lastPoseData = data;
        const debugOutput = document.getElementById('debug-output');
        debugOutput.textContent = JSON.stringify(data, null, 2);
    }

    /**
     * Update status message
     */
    updateStatus(message, type) {
        const statusEl = document.getElementById('connection-status');
        statusEl.textContent = message;
        statusEl.className = type === 'success' || type === 'info' ? 'connected' : 'disconnected';
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    /**
     * Update WebSocket status
     */
    updateWSStatus(message, type) {
        const wsStatusEl = document.getElementById('ws-status');
        wsStatusEl.textContent = message;
        wsStatusEl.className = type === 'connected' ? 'connected' : 'disconnected';
    }

    /**
     * Disconnect from server and end session
     */
    disconnect() {
        if (this.xrSession) {
            this.xrSession.end();
        }

        if (this.ws) {
            this.ws.close();
        }

        this.connected = false;
        this.updateStatus('Disconnected', 'info');
        this.updateWSStatus('Disconnected', 'disconnected');

        document.getElementById('vr-btn').disabled = true;
        document.getElementById('connect-btn').disabled = false;
        document.getElementById('disconnect-btn').disabled = true;
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    const app = new ControllerTracker();

    // Make app globally accessible for debugging
    window.tracker = app;
});
