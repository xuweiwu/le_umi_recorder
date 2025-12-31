"""
Data models for Quest Controller Tracking Client
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class ControllerState:
    """Represents the state of a single controller"""
    position: Tuple[float, float, float]
    """Controller position in 3D space (x, y, z) in meters"""

    orientation: Tuple[float, float, float, float]
    """Controller orientation as quaternion (x, y, z, w)"""

    buttons: Dict[str, Dict[str, any]]
    """Button states, indexed by button number"""

    timestamp: float
    """Timestamp when this state was captured"""

    @classmethod
    def from_dict(cls, data: dict) -> 'ControllerState':
        """Create ControllerState from dictionary"""
        return cls(
            position=tuple(data['position']),
            orientation=tuple(data['orientation']),
            buttons=data.get('buttons', {}),
            timestamp=data['timestamp']
        )

    def is_button_pressed(self, button_index: int) -> bool:
        """Check if a specific button is pressed"""
        button = self.buttons.get(str(button_index), {})
        return button.get('pressed', False)

    def get_button_value(self, button_index: int) -> float:
        """Get analog value of a button (0.0 to 1.0)"""
        button = self.buttons.get(str(button_index), {})
        return button.get('value', 0.0)


@dataclass
class PoseData:
    """Complete pose data for one or both controllers"""
    timestamp: float
    """Timestamp of the pose frame"""

    coordinate_system: str
    """Coordinate system: 'local' or 'world'"""

    left: Optional[ControllerState]
    """Left controller state (None if not tracked)"""

    right: Optional[ControllerState]
    """Right controller state (None if not tracked)"""

    received_at: float
    """Server timestamp when pose was received"""

    @classmethod
    def from_dict(cls, data: dict) -> 'PoseData':
        """Create PoseData from dictionary"""
        controllers = data.get('controllers', {})
        return cls(
            timestamp=data['timestamp'],
            coordinate_system=data.get('coordinate_system', 'local'),
            left=ControllerState.from_dict(controllers['left']) if 'left' in controllers else None,
            right=ControllerState.from_dict(controllers['right']) if 'right' in controllers else None,
            received_at=data.get('received_at', data['timestamp'])
        )

    @property
    def latency(self) -> float:
        """Calculate latency in milliseconds"""
        return (self.received_at - self.timestamp) * 1000


@dataclass
class ServerStatus:
    """Server status information"""
    status: str
    uptime_seconds: float
    quest_clients: int
    visualizer_clients: int
    total_frames_received: int
    current_frame_rate: float
    has_pose_data: bool
    last_update: Optional[float]

    @classmethod
    def from_dict(cls, data: dict) -> 'ServerStatus':
        """Create ServerStatus from dictionary"""
        return cls(
            status=data['status'],
            uptime_seconds=data['uptime_seconds'],
            quest_clients=data['quest_clients'],
            visualizer_clients=data['visualizer_clients'],
            total_frames_received=data['total_frames_received'],
            current_frame_rate=data['current_frame_rate'],
            has_pose_data=data['has_pose_data'],
            last_update=data.get('last_update')
        )
