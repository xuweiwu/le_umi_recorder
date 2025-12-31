"""
Quest Controller Client - Python library for accessing Quest 3 controller tracking data

This package provides both async and sync clients for easy access to the tracking server.

Quick Start (Sync):
    >>> from quest_controller_client import QuestControllerClientSync
    >>> client = QuestControllerClientSync('http://localhost:8000')
    >>> pose = client.get_latest_pose()
    >>> print(pose.left.position if pose.left else "No data")
    >>> client.close()

Quick Start (Async):
    >>> from quest_controller_client import QuestControllerClient
    >>> async def main():
    ...     async with QuestControllerClient('http://localhost:8000') as client:
    ...         pose = await client.get_latest_pose()
    ...         print(pose.left.position if pose.left else "No data")
"""

__version__ = '1.0.0'
__author__ = 'Quest Controller Tracking Project'
__license__ = 'MIT'

# Import main classes
from .client import QuestControllerClient
from .sync_client import QuestControllerClientSync, get_controller_pose
from .models import ControllerState, PoseData, ServerStatus

# Define public API
__all__ = [
    'QuestControllerClient',
    'QuestControllerClientSync',
    'get_controller_pose',
    'ControllerState',
    'PoseData',
    'ServerStatus',
]
