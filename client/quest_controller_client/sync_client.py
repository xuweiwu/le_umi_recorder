"""
Synchronous client wrapper for Quest Controller Tracking Server
Simpler API for users who don't want to deal with async/await
"""

import asyncio
import threading
from typing import Optional, Callable
from queue import Queue, Empty

from .client import QuestControllerClient
from .models import ControllerState, PoseData, ServerStatus


class QuestControllerClientSync:
    """
    Synchronous wrapper for QuestControllerClient

    Easier to use for simple scripts and interactive use.

    Example:
        >>> client = QuestControllerClientSync('http://localhost:8000')
        >>> pose = client.get_latest_pose()
        >>> print(pose.left.position if pose.left else "No data")
        >>> client.close()
    """

    def __init__(self, base_url: str = 'http://localhost:8000'):
        """
        Initialize the synchronous client

        Args:
            base_url: Base URL of the tracking server
        """
        self.base_url = base_url
        self._loop = None
        self._thread = None
        self._client = None
        self._running = False
        self._start_event_loop()

    def _start_event_loop(self):
        """Start background event loop in a thread"""
        self._loop = asyncio.new_event_loop()
        self._running = True

        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Create async client in the loop
        future = asyncio.run_coroutine_threadsafe(
            self._create_client(),
            self._loop
        )
        future.result()

    async def _create_client(self):
        """Create the async client"""
        self._client = QuestControllerClient(self.base_url)

    def _run_async(self, coro):
        """Run a coroutine in the background loop and wait for result"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        """Close the client and cleanup"""
        if self._client:
            self._run_async(self._client.close())

        if self._loop and self._running:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._running = False

        if self._thread:
            self._thread.join(timeout=1)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Cleanup on deletion"""
        if self._running:
            self.close()

    # Status Methods

    def get_status(self) -> ServerStatus:
        """
        Get server status

        Returns:
            ServerStatus object

        Example:
            >>> status = client.get_status()
            >>> print(f"Frame rate: {status.current_frame_rate} Hz")
        """
        return self._run_async(self._client.get_status())

    def is_connected(self) -> bool:
        """
        Check if server is reachable

        Returns:
            True if connected, False otherwise
        """
        return self._run_async(self._client.is_connected())

    # Pose Query Methods

    def get_latest_pose(self) -> Optional[PoseData]:
        """
        Get the latest controller pose

        Returns:
            PoseData if available, None otherwise

        Example:
            >>> pose = client.get_latest_pose()
            >>> if pose and pose.left:
            ...     print(f"Left position: {pose.left.position}")
        """
        return self._run_async(self._client.get_latest_pose())

    def get_left_controller(self) -> Optional[ControllerState]:
        """
        Get left controller state

        Returns:
            ControllerState if available, None otherwise
        """
        return self._run_async(self._client.get_left_controller())

    def get_right_controller(self) -> Optional[ControllerState]:
        """
        Get right controller state

        Returns:
            ControllerState if available, None otherwise
        """
        return self._run_async(self._client.get_right_controller())

    # Streaming Methods

    def stream(
        self,
        callback: Callable[[PoseData], None],
        blocking: bool = True,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Stream real-time pose updates

        Args:
            callback: Function to call with each pose
            blocking: If True, blocks until connection closes
            on_connect: Optional callback when connected
            on_disconnect: Optional callback when disconnected
            on_error: Optional callback for errors

        Example:
            >>> def on_pose(pose):
            ...     print(f"Frame: {pose.timestamp}")
            >>> client.stream(on_pose, blocking=True)
        """
        async def stream_task():
            await self._client.stream(
                callback,
                on_connect=on_connect,
                on_disconnect=on_disconnect,
                on_error=on_error
            )

        if blocking:
            self._run_async(stream_task())
        else:
            asyncio.run_coroutine_threadsafe(stream_task(), self._loop)

    def poll(
        self,
        callback: Callable[[PoseData], None],
        rate_hz: float = 10.0,
        duration: Optional[float] = None
    ):
        """
        Poll for pose updates at a fixed rate

        Args:
            callback: Function to call with each pose
            rate_hz: Polling rate in Hz
            duration: Duration in seconds (None = infinite)

        Example:
            >>> def on_pose(pose):
            ...     print(pose.left.position if pose.left else "N/A")
            >>> client.poll(on_pose, rate_hz=5, duration=10)
        """
        self._run_async(self._client.poll(callback, rate_hz, duration))

    # Iterator interface for easy loops

    def iter_poses(self, rate_hz: float = 10.0, timeout: float = 1.0):
        """
        Iterate over poses at a fixed rate

        Args:
            rate_hz: Polling rate in Hz
            timeout: Timeout for each poll

        Yields:
            PoseData objects

        Example:
            >>> for pose in client.iter_poses(rate_hz=5):
            ...     if pose and pose.left:
            ...         print(pose.left.position)
            ...     if some_condition:
            ...         break
        """
        import time

        interval = 1.0 / rate_hz

        try:
            while True:
                pose = self.get_latest_pose()
                if pose:
                    yield pose

                time.sleep(interval)

        except KeyboardInterrupt:
            pass


# Convenience function for quick access

def get_controller_pose(
    base_url: str = 'http://localhost:8000',
    controller: str = 'latest'
) -> Optional[PoseData | ControllerState]:
    """
    Quick function to get a single pose without managing client lifecycle

    Args:
        base_url: Server URL
        controller: 'latest', 'left', or 'right'

    Returns:
        Pose data or None

    Example:
        >>> pose = get_controller_pose()
        >>> left = get_controller_pose(controller='left')
    """
    with QuestControllerClientSync(base_url) as client:
        if controller == 'latest':
            return client.get_latest_pose()
        elif controller == 'left':
            return client.get_left_controller()
        elif controller == 'right':
            return client.get_right_controller()
        else:
            raise ValueError(f"Invalid controller: {controller}")
