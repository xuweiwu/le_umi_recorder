# WebXR Controller Tracking

Stream Meta Quest 3 controller poses to Python at 30+ Hz with real-time Rerun visualization.

## Architecture

```
Quest 3 Browser ──WebSocket──> Python Server ──> Rerun Visualizer
    (WebXR)                    (aiohttp)         (3D Display)
                                   │
                                   └──> REST API / Custom Apps
```

## Quick Start

```bash
# Start server (auto-setup with SSL)
./start_server.sh

# On Quest 3 browser: https://<your-ip>:8000
# Accept SSL warning, click "Enter VR"
```

**Manual setup:**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python generate_cert.py  # Required for WebXR
python server.py
```

## Python Client

```bash
cd client && pip install -e .
```

```python
from quest_controller_client import QuestControllerClientSync

with QuestControllerClientSync('https://localhost:8000') as client:
    for pose in client:
        if pose.left:
            print(f"Left: {pose.left.position}")
```

## REST API

```bash
curl https://localhost:8000/api/pose/latest
curl https://localhost:8000/api/status
```

## UMI Dataset Recording

Record controller poses + camera in LeRobot format for robot learning.

### Setup

```bash
cd umi && pip install -r requirements.txt
```

### Recording

```bash
# List cameras
python -m umi.lerobot_recorder --list-cameras

# Using config file (recommended)
cp umi/config.example.yaml config.yaml  # Edit with your settings
python -m umi.lerobot_recorder --config config.yaml

# Or with CLI args
python -m umi.lerobot_recorder --repo-id user/dataset --camera 0:wrist:640x480

# With robot URDF (IK visualization)
python -m umi.lerobot_recorder --config config.yaml --urdf /path/to/robot.urdf
```

**Config file** (`config.yaml`):
```yaml
repo_id: user/my_dataset
camera: 0:wrist:640x480
hand: right
fps: 30
urdf: /path/to/robot.urdf  # optional
tasks:
  - Pick up the cup
  - Place on shelf
```

**Controls:** `1-9` select task, `s` start, `e` end episode, `q` quit

### Playback

```bash
python -m umi.visualize_dataset --dataset ./datasets/my_dataset --list
python -m umi.visualize_dataset --dataset ./datasets/my_dataset --episode 0
```

### Dataset Format

```
datasets/my_dataset/
├── meta/info.json, episodes.jsonl
├── data/episode_*.parquet
└── videos/observation.images.*/episode_*.mp4
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| WebXR not working | Use HTTPS (`https://`), accept SSL warning |
| Can't connect | Check firewall, verify same network |
| No controller data | Enter VR mode, grant permissions |
| Low FPS | Use 5GHz WiFi, reduce update rate |

## Requirements

- Meta Quest 3
- Python 3.8+
- Same network for Quest and server
