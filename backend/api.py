"""
Python API Client for Controller Tracking Server
Easy-to-use interface for accessing controller pose data
"""

import asyncio
import json
import ssl
import time
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass

import aiohttp
import websockets


@dataclass
class ControllerState:
    """Represents the state of a single controller"""
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # quaternion (x, y, z, w)
    buttons: dict
    timestamp: float

    @classmethod
    def from_dict(cls, data: dict) -> 'ControllerState':
        """Create from dictionary"""
        return cls(
            position=tuple(data['position']),
            orientation=tuple(data['orientation']),
            buttons=data.get('buttons', {}),
            timestamp=data['timestamp']
        )


@dataclass
class PoseData:
    """Complete pose data for controllers"""
    timestamp: float
    coordinate_system: str
    left: Optional[ControllerState]
    right: Optional[ControllerState]
    received_at: float

    @classmethod
    def from_dict(cls, data: dict) -> 'PoseData':
        """Create from dictionary"""
        controllers = data.get('controllers', {})
        return cls(
            timestamp=data['timestamp'],
            coordinate_system=data.get('coordinate_system', 'local'),
            left=ControllerState.from_dict(controllers['left']) if 'left' in controllers else None,
            right=ControllerState.from_dict(controllers['right']) if 'right' in controllers else None,
            received_at=data.get('received_at', time.time())
        )


class ControllerTrackingClient:
    """
    Client for accessing controller tracking data

    Usage:
        # REST API (polling)
        client = ControllerTrackingClient('http://localhost:8000')
        pose = await client.get_latest_pose()

        # WebSocket (streaming)
        client = ControllerTrackingClient('http://localhost:8000')
        await client.connect_stream(callback)
    """

    def __init__(self, base_url: str = 'http://localhost:8000', verify_ssl: bool = False):
        self.base_url = base_url.rstrip('/')
        self.ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

        # Create SSL context for self-signed certificates
        self.ssl_context = None
        if not verify_ssl:
            self.ssl_context = ssl.create_default_context()
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)

    async def close(self):
        """Close connections"""
        if self.session and not self.session.closed:
            await self.session.close()
        if self.ws:
            await self.ws.close()

    async def get_status(self) -> dict:
        """Get server status"""
        await self._ensure_session()
        async with self.session.get(f'{self.base_url}/api/status') as resp:
            return await resp.json()

    async def get_latest_pose(self) -> Optional[PoseData]:
        """Get latest controller pose data"""
        await self._ensure_session()
        try:
            async with self.session.get(f'{self.base_url}/api/pose/latest') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return PoseData.from_dict(data)
                return None
        except Exception as e:
            print(f'Error getting pose: {e}')
            return None

    async def get_left_controller(self) -> Optional[ControllerState]:
        """Get left controller state"""
        await self._ensure_session()
        try:
            async with self.session.get(f'{self.base_url}/api/pose/left') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return ControllerState.from_dict(data)
                return None
        except Exception:
            return None

    async def get_right_controller(self) -> Optional[ControllerState]:
        """Get right controller state"""
        await self._ensure_session()
        try:
            async with self.session.get(f'{self.base_url}/api/pose/right') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return ControllerState.from_dict(data)
                return None
        except Exception:
            return None

    async def connect_stream(self, callback: Callable[[PoseData], None]):
        """
        Connect to WebSocket stream and receive real-time pose updates

        Args:
            callback: Function to call with each new pose update
        """
        try:
            # Only use SSL context for wss:// URLs
            ssl_arg = self.ssl_context if self.ws_url.startswith('wss://') else None
            async with websockets.connect(self.ws_url, ssl=ssl_arg) as websocket:
                # Send handshake
                await websocket.send(json.dumps({
                    'type': 'handshake',
                    'client': 'visualizer',
                    'timestamp': time.time()
                }))

                # Wait for acknowledgment
                ack = await websocket.recv()
                print(f'Connected: {ack}')

                # Receive pose updates
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'controllers' in data:
                            pose = PoseData.from_dict(data)
                            callback(pose)
                    except json.JSONDecodeError:
                        print(f'Invalid JSON: {message}')
                    except Exception as e:
                        print(f'Error processing message: {e}')

        except Exception as e:
            print(f'WebSocket error: {e}')


# Example usage functions

async def example_polling():
    """Example: Poll for controller data"""
    client = ControllerTrackingClient('http://localhost:8000')

    try:
        # Get server status
        status = await client.get_status()
        print(f'Server status: {status}')

        # Poll for poses
        for _ in range(10):
            pose = await client.get_latest_pose()
            if pose:
                print(f'\nTimestamp: {pose.timestamp}')
                if pose.left:
                    print(f'Left: pos={pose.left.position}, ori={pose.left.orientation}')
                if pose.right:
                    print(f'Right: pos={pose.right.position}, ori={pose.right.orientation}')
            else:
                print('No pose data available')

            await asyncio.sleep(0.1)

    finally:
        await client.close()


async def example_streaming():
    """Example: Stream controller data via WebSocket"""
    client = ControllerTrackingClient('http://localhost:8000')

    def on_pose_update(pose: PoseData):
        """Callback for each pose update"""
        print(f'\nPose update at {pose.timestamp}')
        if pose.left:
            print(f'  Left: {pose.left.position}')
        if pose.right:
            print(f'  Right: {pose.right.position}')

    try:
        await client.connect_stream(on_pose_update)
    finally:
        await client.close()


if __name__ == '__main__':
    # Run polling example
    print('Running polling example...')
    asyncio.run(example_polling())

    # Run streaming example
    # print('Running streaming example...')
    # asyncio.run(example_streaming())
