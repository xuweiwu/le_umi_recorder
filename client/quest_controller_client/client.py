"""
Async client for Quest Controller Tracking Server
"""

import asyncio
import json
import time
from typing import Optional, Callable, Awaitable

import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol

from .models import ControllerState, PoseData, ServerStatus


class QuestControllerClient:
    """
    Async client for accessing Quest controller tracking data

    Example:
        >>> async def main():
        ...     client = QuestControllerClient('http://localhost:8000')
        ...     pose = await client.get_latest_pose()
        ...     print(pose.left.position if pose.left else "No left controller")
        ...     await client.close()
    """

    def __init__(self, base_url: str = 'http://localhost:8000'):
        """
        Initialize the client

        Args:
            base_url: Base URL of the tracking server (e.g., 'http://192.168.1.100:8000')
        """
        self.base_url = base_url.rstrip('/')
        self.ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[WebSocketClientProtocol] = None

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        """Close all connections"""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._ws and not self._ws.closed:
            await self._ws.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    # Status and Info Methods

    async def get_status(self) -> ServerStatus:
        """
        Get server status

        Returns:
            ServerStatus object with server information

        Raises:
            aiohttp.ClientError: If connection fails
        """
        await self._ensure_session()
        async with self._session.get(f'{self.base_url}/api/status') as resp:
            resp.raise_for_status()
            data = await resp.json()
            return ServerStatus.from_dict(data)

    async def is_connected(self) -> bool:
        """
        Check if server is reachable

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            await self.get_status()
            return True
        except Exception:
            return False

    # Pose Query Methods (REST API)

    async def get_latest_pose(self) -> Optional[PoseData]:
        """
        Get the latest controller pose data

        Returns:
            PoseData object if available, None otherwise
        """
        await self._ensure_session()
        try:
            async with self._session.get(f'{self.base_url}/api/pose/latest') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return PoseData.from_dict(data)
                return None
        except Exception:
            return None

    async def get_left_controller(self) -> Optional[ControllerState]:
        """
        Get left controller state

        Returns:
            ControllerState if available, None otherwise
        """
        await self._ensure_session()
        try:
            async with self._session.get(f'{self.base_url}/api/pose/left') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return ControllerState.from_dict(data)
                return None
        except Exception:
            return None

    async def get_right_controller(self) -> Optional[ControllerState]:
        """
        Get right controller state

        Returns:
            ControllerState if available, None otherwise
        """
        await self._ensure_session()
        try:
            async with self._session.get(f'{self.base_url}/api/pose/right') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return ControllerState.from_dict(data)
                return None
        except Exception:
            return None

    # Streaming Methods (WebSocket)

    async def stream(
        self,
        callback: Callable[[PoseData], None],
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Stream real-time pose updates via WebSocket

        Args:
            callback: Function to call with each pose update
            on_connect: Optional callback when connection is established
            on_disconnect: Optional callback when connection closes
            on_error: Optional callback for errors

        Example:
            >>> def on_pose(pose: PoseData):
            ...     print(f"Position: {pose.left.position if pose.left else 'N/A'}")
            >>> await client.stream(on_pose)
        """
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self._ws = websocket

                # Send handshake
                await websocket.send(json.dumps({
                    'type': 'handshake',
                    'client': 'visualizer',
                    'timestamp': time.time()
                }))

                # Wait for acknowledgment
                ack_msg = await websocket.recv()
                if on_connect:
                    on_connect()

                # Receive pose updates
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'controllers' in data:
                            pose = PoseData.from_dict(data)
                            callback(pose)
                    except json.JSONDecodeError as e:
                        if on_error:
                            on_error(e)
                    except Exception as e:
                        if on_error:
                            on_error(e)

        except Exception as e:
            if on_error:
                on_error(e)
        finally:
            self._ws = None
            if on_disconnect:
                on_disconnect()

    async def stream_async(
        self,
        callback: Callable[[PoseData], Awaitable[None]],
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None
    ):
        """
        Stream real-time pose updates via WebSocket (async callback version)

        Args:
            callback: Async function to call with each pose update
            on_connect: Optional async callback when connection is established
            on_disconnect: Optional async callback when connection closes
            on_error: Optional async callback for errors

        Example:
            >>> async def on_pose(pose: PoseData):
            ...     await save_to_database(pose)
            >>> await client.stream_async(on_pose)
        """
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self._ws = websocket

                # Send handshake
                await websocket.send(json.dumps({
                    'type': 'handshake',
                    'client': 'visualizer',
                    'timestamp': time.time()
                }))

                # Wait for acknowledgment
                ack_msg = await websocket.recv()
                if on_connect:
                    await on_connect()

                # Receive pose updates
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'controllers' in data:
                            pose = PoseData.from_dict(data)
                            await callback(pose)
                    except json.JSONDecodeError as e:
                        if on_error:
                            await on_error(e)
                    except Exception as e:
                        if on_error:
                            await on_error(e)

        except Exception as e:
            if on_error:
                await on_error(e)
        finally:
            self._ws = None
            if on_disconnect:
                await on_disconnect()

    # Polling Methods

    async def poll(
        self,
        callback: Callable[[PoseData], None],
        rate_hz: float = 10.0,
        duration: Optional[float] = None
    ):
        """
        Poll for pose updates at a fixed rate

        Args:
            callback: Function to call with each pose
            rate_hz: Polling rate in Hz (default: 10)
            duration: Optional duration in seconds (None = infinite)

        Example:
            >>> def on_pose(pose: PoseData):
            ...     print(pose.left.position if pose.left else "N/A")
            >>> await client.poll(on_pose, rate_hz=5, duration=30)
        """
        interval = 1.0 / rate_hz
        start_time = time.time()

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break

                # Get pose
                pose = await self.get_latest_pose()
                if pose:
                    callback(pose)

                # Wait
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            pass
