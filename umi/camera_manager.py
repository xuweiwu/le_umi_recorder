"""
Camera Manager for UMI Recording
Handles multiple USB cameras with synchronized capture
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    name: str                    # e.g., "camera_0", "wrist_left"
    index: int                   # OpenCV camera index (0, 1, 2...)
    width: int = 640
    height: int = 480
    fps: int = 30

    @classmethod
    def from_string(cls, config_str: str) -> 'CameraConfig':
        """
        Parse camera config from string format: INDEX:NAME:WxH
        Examples: "0:camera_0:640x480", "1:wrist:1280x720"
        """
        parts = config_str.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid camera config: {config_str}. Expected INDEX:NAME[:WxH]")

        index = int(parts[0])
        name = parts[1]
        width, height = 640, 480

        if len(parts) >= 3:
            dims = parts[2].lower().split('x')
            if len(dims) == 2:
                width, height = int(dims[0]), int(dims[1])

        return cls(name=name, index=index, width=width, height=height)


class CameraManager:
    """
    Manages multiple USB cameras for synchronized capture.

    Usage:
        configs = [CameraConfig("camera_0", 0), CameraConfig("camera_1", 1)]
        manager = CameraManager(configs)
        manager.connect_all()

        frames = manager.capture_all()  # {"camera_0": np.ndarray, "camera_1": np.ndarray}

        manager.disconnect_all()
    """

    def __init__(self, configs: List[CameraConfig]):
        self.configs = configs
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self._connected = False

    def connect_all(self) -> bool:
        """
        Connect to all configured cameras.

        Returns:
            True if all cameras connected successfully
        """
        if self._connected:
            return True

        success = True
        for config in self.configs:
            cap = cv2.VideoCapture(config.index)

            if not cap.isOpened():
                print(f"Warning: Could not open camera {config.name} (index {config.index})")
                success = False
                continue

            # Configure camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(cv2.CAP_PROP_FPS, config.fps)

            # Verify settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera {config.name}: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            self.cameras[config.name] = cap

        self._connected = len(self.cameras) > 0
        return success

    def disconnect_all(self):
        """Release all camera resources."""
        for name, cap in self.cameras.items():
            cap.release()
            print(f"Disconnected camera: {name}")

        self.cameras.clear()
        self._connected = False

    def capture_all(self, timeout_ms: int = 100) -> Dict[str, np.ndarray]:
        """
        Capture frames from all cameras as synchronously as possible.

        Args:
            timeout_ms: Maximum time to wait for a frame (not currently used, for future)

        Returns:
            Dictionary mapping camera name to BGR image (numpy array)
        """
        frames = {}

        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR to RGB for LeRobot compatibility
                frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print(f"Warning: Failed to capture from camera {name}")

        return frames

    def capture_single(self, camera_name: str) -> Optional[np.ndarray]:
        """Capture a single frame from a specific camera."""
        if camera_name not in self.cameras:
            return None

        ret, frame = self.cameras[camera_name].read()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_camera_info(self) -> Dict[str, dict]:
        """Get information about connected cameras."""
        info = {}
        for name, cap in self.cameras.items():
            info[name] = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
            }
        return info

    def get_feature_definitions(self) -> Dict[str, dict]:
        """
        Return LeRobot feature definitions for all cameras.

        Returns:
            Dictionary of feature definitions suitable for LeRobotDataset
        """
        features = {}
        for config in self.configs:
            if config.name in self.cameras:
                features[f"observation.images.{config.name}"] = {
                    "dtype": "video",
                    "shape": (config.height, config.width, 3),
                    "names": ["height", "width", "channel"],
                }
        return features

    @property
    def is_connected(self) -> bool:
        """Check if any cameras are connected."""
        return self._connected and len(self.cameras) > 0

    @property
    def camera_names(self) -> List[str]:
        """Get list of connected camera names."""
        return list(self.cameras.keys())

    @staticmethod
    def list_available_cameras(max_index: int = 10) -> List[int]:
        """
        Discover available camera indices.

        Args:
            max_index: Maximum index to check

        Returns:
            List of available camera indices
        """
        available = []

        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()

        return available

    def __enter__(self):
        """Context manager entry."""
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_all()


def main():
    """Test camera discovery and capture."""
    print("Discovering available cameras...")
    available = CameraManager.list_available_cameras()
    print(f"Found cameras at indices: {available}")

    if not available:
        print("No cameras found!")
        return

    # Test first camera
    config = CameraConfig(name="test_camera", index=available[0])
    manager = CameraManager([config])

    with manager:
        print(f"\nCamera info: {manager.get_camera_info()}")
        print("\nCapturing 10 frames...")

        for i in range(10):
            frames = manager.capture_all()
            if frames:
                frame = frames["test_camera"]
                print(f"Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            time.sleep(0.1)

    print("\nDone!")


if __name__ == '__main__':
    main()
