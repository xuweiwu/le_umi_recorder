# Quest Controller Client

Python client library for accessing Meta Quest 3 controller tracking data.

## Installation

### From source

```bash
cd client
pip install -e .
```

### With pip (when published)

```bash
pip install quest-controller-client
```

## Quick Start

### Synchronous API (Easiest)

For simple scripts and interactive use:

```python
from quest_controller_client import QuestControllerClientSync

# Create client
client = QuestControllerClientSync('http://localhost:8000')

# Get latest pose
pose = client.get_latest_pose()
if pose and pose.left:
    print(f"Left controller: {pose.left.position}")

# Close when done
client.close()
```

Or use context manager:

```python
from quest_controller_client import QuestControllerClientSync

with QuestControllerClientSync('http://localhost:8000') as client:
    pose = client.get_latest_pose()
    if pose:
        print(f"Timestamp: {pose.timestamp}")
        if pose.left:
            print(f"Left: {pose.left.position}")
        if pose.right:
            print(f"Right: {pose.right.position}")
```

### Async API (High Performance)

For high-performance applications:

```python
import asyncio
from quest_controller_client import QuestControllerClient

async def main():
    async with QuestControllerClient('http://localhost:8000') as client:
        # Get status
        status = await client.get_status()
        print(f"Server running at {status.current_frame_rate} Hz")

        # Get pose
        pose = await client.get_latest_pose()
        if pose and pose.left:
            print(f"Left controller: {pose.left.position}")

asyncio.run(main())
```

## Usage Examples

### 1. Stream Real-Time Data (Sync)

```python
from quest_controller_client import QuestControllerClientSync

def on_pose(pose):
    if pose.left:
        print(f"Left: {pose.left.position}")
    if pose.right:
        print(f"Right: {pose.right.position}")

with QuestControllerClientSync('http://localhost:8000') as client:
    client.stream(on_pose, blocking=True)
```

### 2. Stream Real-Time Data (Async)

```python
import asyncio
from quest_controller_client import QuestControllerClient

async def main():
    client = QuestControllerClient('http://localhost:8000')

    def on_pose(pose):
        print(f"Frame {pose.timestamp}: {pose.left.position if pose.left else 'N/A'}")

    await client.stream(on_pose)

asyncio.run(main())
```

### 3. Poll at Fixed Rate

```python
from quest_controller_client import QuestControllerClientSync

def on_pose(pose):
    if pose:
        print(f"Left: {pose.left.position if pose.left else 'N/A'}")

with QuestControllerClientSync('http://localhost:8000') as client:
    # Poll at 5 Hz for 10 seconds
    client.poll(on_pose, rate_hz=5, duration=10)
```

### 4. Iterate Over Poses

```python
from quest_controller_client import QuestControllerClientSync

with QuestControllerClientSync('http://localhost:8000') as client:
    for pose in client.iter_poses(rate_hz=10):
        if pose and pose.left:
            x, y, z = pose.left.position
            print(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")

            # Exit on some condition
            if some_condition:
                break
```

### 5. Quick One-Shot Query

```python
from quest_controller_client import get_controller_pose

# Get latest pose
pose = get_controller_pose('http://localhost:8000')

# Get specific controller
left = get_controller_pose('http://localhost:8000', controller='left')
right = get_controller_pose('http://localhost:8000', controller='right')
```

### 6. Check Button States

```python
from quest_controller_client import QuestControllerClientSync

with QuestControllerClientSync('http://localhost:8000') as client:
    pose = client.get_latest_pose()

    if pose and pose.left:
        # Check if trigger is pressed (button 0)
        if pose.left.is_button_pressed(0):
            print("Trigger pressed!")

        # Get grip value (button 1)
        grip_value = pose.left.get_button_value(1)
        print(f"Grip: {grip_value:.2f}")
```

### 7. Async Streaming with Async Callback

```python
import asyncio
from quest_controller_client import QuestControllerClient

async def main():
    client = QuestControllerClient('http://localhost:8000')

    async def on_pose(pose):
        # Do async operations
        await save_to_database(pose)
        await send_to_mqtt(pose)

    await client.stream_async(on_pose)

asyncio.run(main())
```

### 8. Monitor Server Status

```python
from quest_controller_client import QuestControllerClientSync

with QuestControllerClientSync('http://localhost:8000') as client:
    status = client.get_status()

    print(f"Status: {status.status}")
    print(f"Uptime: {status.uptime_seconds:.1f}s")
    print(f"Quest clients: {status.quest_clients}")
    print(f"Frame rate: {status.current_frame_rate} Hz")
    print(f"Total frames: {status.total_frames_received}")
    print(f"Has data: {status.has_pose_data}")
```

### 9. Connection Callbacks

```python
from quest_controller_client import QuestControllerClientSync

def on_connect():
    print("Connected to server!")

def on_disconnect():
    print("Disconnected from server")

def on_error(error):
    print(f"Error: {error}")

def on_pose(pose):
    print(f"Pose: {pose.timestamp}")

with QuestControllerClientSync('http://localhost:8000') as client:
    client.stream(
        on_pose,
        on_connect=on_connect,
        on_disconnect=on_disconnect,
        on_error=on_error,
        blocking=True
    )
```

### 10. Data Analysis

```python
import time
from quest_controller_client import QuestControllerClientSync

positions = []

def collect_position(pose):
    if pose and pose.left:
        positions.append(pose.left.position)

with QuestControllerClientSync('http://localhost:8000') as client:
    # Collect data for 5 seconds at 30 Hz
    client.poll(collect_position, rate_hz=30, duration=5)

# Analyze
print(f"Collected {len(positions)} positions")
avg_x = sum(p[0] for p in positions) / len(positions)
print(f"Average X: {avg_x:.3f}")
```

## API Reference

### QuestControllerClientSync

Synchronous client (easiest to use).

**Methods:**

- `get_status() -> ServerStatus` - Get server status
- `is_connected() -> bool` - Check if server is reachable
- `get_latest_pose() -> Optional[PoseData]` - Get latest pose
- `get_left_controller() -> Optional[ControllerState]` - Get left controller
- `get_right_controller() -> Optional[ControllerState]` - Get right controller
- `stream(callback, blocking=True, ...)` - Stream real-time data
- `poll(callback, rate_hz=10, duration=None)` - Poll at fixed rate
- `iter_poses(rate_hz=10)` - Iterate over poses
- `close()` - Close connections

### QuestControllerClient

Async client (high performance).

**Methods:**

- `async get_status() -> ServerStatus`
- `async is_connected() -> bool`
- `async get_latest_pose() -> Optional[PoseData]`
- `async get_left_controller() -> Optional[ControllerState]`
- `async get_right_controller() -> Optional[ControllerState]`
- `async stream(callback, ...)` - Stream with sync callback
- `async stream_async(async_callback, ...)` - Stream with async callback
- `async poll(callback, rate_hz=10, duration=None)`
- `async close()`

### Data Models

**PoseData:**
- `timestamp: float` - Pose timestamp
- `coordinate_system: str` - 'local' or 'world'
- `left: Optional[ControllerState]` - Left controller
- `right: Optional[ControllerState]` - Right controller
- `received_at: float` - Server receipt time
- `latency: float` - Latency in milliseconds (property)

**ControllerState:**
- `position: Tuple[float, float, float]` - Position (x, y, z) in meters
- `orientation: Tuple[float, float, float, float]` - Quaternion (x, y, z, w)
- `buttons: Dict` - Button states
- `timestamp: float` - Timestamp
- `is_button_pressed(index: int) -> bool` - Check button
- `get_button_value(index: int) -> float` - Get button value (0-1)

**ServerStatus:**
- `status: str` - Server status
- `uptime_seconds: float` - Uptime
- `quest_clients: int` - Connected Quest clients
- `visualizer_clients: int` - Connected visualizers
- `total_frames_received: int` - Total frames
- `current_frame_rate: float` - Current frame rate (Hz)
- `has_pose_data: bool` - Has pose data available
- `last_update: Optional[float]` - Last update timestamp

## Coordinate Systems

- **Local**: Relative to initial VR position (recommended)
- **World**: Relative to Quest's world origin (may shift)

Toggle in the Quest web interface.

## Performance

### Sync vs Async

- **Sync**: Easier to use, good for most cases, ~10-30 Hz
- **Async**: Better performance, required for 60+ Hz

### Streaming vs Polling

- **Streaming**: Real-time WebSocket, 30-90 Hz, low latency
- **Polling**: REST API, up to ~10 Hz, simpler

## Requirements

- Python 3.8+
- aiohttp >= 3.9.0
- websockets >= 12.0

## License

MIT License

## Links

- [Full Project Documentation](../README.md)
- [Setup Guide](../SETUP.md)
- [Architecture](../ARCHITECTURE.md)
