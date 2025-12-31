# Examples

Example scripts demonstrating how to use the Controller Tracking API.

## Setup

All examples require the backend server to be running:

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

## Available Examples

### 1. Simple Collector (`simple_collector.py`)

Basic real-time streaming of controller positions.

**Usage:**
```bash
python simple_collector.py
```

**Output:**
```
Frame 123.45 | LEFT: ( 0.123,  0.456,  0.789) | RIGHT: (-0.123,  0.456,  0.789)
```

### 2. CSV Logger (`csv_logger.py`)

Log controller data to CSV file for analysis.

**Usage:**
```bash
# Log indefinitely (Ctrl+C to stop)
python csv_logger.py

# Log for 60 seconds
python csv_logger.py --duration 60

# Custom output file
python csv_logger.py --output my_recording.csv

# Connect to remote server
python csv_logger.py --server http://192.168.1.100:8000
```

**CSV Format:**
```csv
frame,timestamp,coordinate_system,hand,pos_x,pos_y,pos_z,ori_x,ori_y,ori_z,ori_w,button_0_pressed,button_1_pressed,received_at
1,1234567890.123,local,left,0.1,0.2,0.3,0,0,0,1,0,0,1234567890.124
1,1234567890.123,local,right,-0.1,0.2,0.3,0,0,0,1,0,0,1234567890.124
```

### 3. Polling Example (`polling_example.py`)

REST API polling instead of WebSocket streaming.

**Usage:**
```bash
python polling_example.py
```

**When to use:**
- Simpler integration
- Lower frequency sampling (< 10 Hz)
- One-off queries
- Firewall restrictions on WebSockets

### 4. Visualizer (in `../visualizer/`)

3D real-time visualization using Rerun.

**Usage:**
```bash
cd ../visualizer
source venv/bin/activate
pip install -r requirements.txt
python visualize.py
```

## Creating Custom Applications

### WebSocket Streaming (Recommended)

For real-time, high-frequency data:

```python
import asyncio
from api import ControllerTrackingClient

async def my_application():
    client = ControllerTrackingClient('http://localhost:8000')

    def on_pose_update(pose):
        # Your processing logic here
        if pose.left:
            print(f'Left: {pose.left.position}')
        if pose.right:
            print(f'Right: {pose.right.position}')

    await client.connect_stream(on_pose_update)

asyncio.run(my_application())
```

### REST API Polling

For simpler use cases:

```python
import asyncio
from api import ControllerTrackingClient

async def my_application():
    client = ControllerTrackingClient('http://localhost:8000')

    try:
        while True:
            pose = await client.get_latest_pose()
            if pose:
                # Your processing logic here
                pass
            await asyncio.sleep(0.1)  # 10 Hz
    finally:
        await client.close()

asyncio.run(my_application())
```

## Data Analysis

After collecting data with `csv_logger.py`, you can analyze it with pandas:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('tracking_data_20240101_120000.csv')

# Filter left hand
left_df = df[df['hand'] == 'left']

# Plot position over time
plt.figure(figsize=(12, 4))
plt.plot(left_df['timestamp'], left_df['pos_x'], label='X')
plt.plot(left_df['timestamp'], left_df['pos_y'], label='Y')
plt.plot(left_df['timestamp'], left_df['pos_z'], label='Z')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.title('Left Controller Position')
plt.show()
```

## Common Use Cases

### Motion Capture

Use the CSV logger to record controller movements:

```bash
python csv_logger.py --duration 30 --output motion_capture.csv
```

### Gesture Recognition

Collect training data for gesture recognition:

```python
# Modify csv_logger.py to label gestures
# Add custom fields for gesture type
```

### VR Input Testing

Test VR input latency and accuracy:

```python
# Compare timestamp vs received_at for latency
# Analyze position jitter for accuracy
```

### Robotics Integration

Stream controller poses to robot control system:

```python
# Use WebSocket streaming for real-time control
# Transform coordinates to robot frame
# Send commands based on controller position
```

## Performance Tips

1. **WebSocket vs Polling**: Use WebSocket for > 10 Hz, polling for < 10 Hz
2. **Processing**: Keep `on_pose_update()` callback fast (< 10ms)
3. **Buffering**: If processing is slow, use a queue
4. **Network**: Use 5GHz WiFi or Ethernet for best latency
5. **Server**: Run on dedicated machine for high-frequency tracking

## Dependencies

All examples use:
- `aiohttp` - HTTP client
- `websockets` - WebSocket client
- Backend `api.py` module

CSV analysis requires:
- `pandas` - Data analysis
- `matplotlib` - Plotting (optional)
