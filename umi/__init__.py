"""
UMI Recording Package
Tools for recording and visualizing controller tracking data in LeRobot format
"""

# Only import base modules that don't have complex dependencies
from .camera_manager import CameraManager, CameraConfig
from .episode_controller import EpisodeController, EpisodeState
from .lerobot_writer import LeRobotFormatWriter, LeRobotFormatReader, FeatureDefinition

# Lazy import for recorder to avoid circular import warning when running as module
def __getattr__(name):
    if name == 'LeRobotDatasetRecorder':
        from .lerobot_recorder import LeRobotDatasetRecorder
        return LeRobotDatasetRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'CameraManager',
    'CameraConfig',
    'EpisodeController',
    'EpisodeState',
    'LeRobotFormatWriter',
    'LeRobotFormatReader',
    'FeatureDefinition',
    'LeRobotDatasetRecorder',
]
