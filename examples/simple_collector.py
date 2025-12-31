"""
Simple Controller Data Collector
Demonstrates basic usage of the tracking API
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from api import ControllerTrackingClient


async def main():
    """Simple example: Print controller positions in real-time"""

    client = ControllerTrackingClient('http://localhost:8000')

    print('Connecting to server...')

    # Check server status
    try:
        status = await client.get_status()
        print(f'\nServer Status:')
        print(f'  Uptime: {status["uptime_seconds"]:.1f}s')
        print(f'  Quest clients: {status["quest_clients"]}')
        print(f'  Frame rate: {status["current_frame_rate"]} Hz')
        print(f'  Total frames: {status["total_frames_received"]}')
        print()
    except Exception as e:
        print(f'Error connecting to server: {e}')
        print('Make sure the server is running on http://localhost:8000')
        return

    # Define callback for pose updates
    def on_pose_update(pose):
        """Called for each new pose frame"""

        # Clear line and print current positions
        output = f'\rFrame {pose.timestamp:.2f} | '

        if pose.left:
            pos = pose.left.position
            output += f'LEFT: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}) | '
        else:
            output += 'LEFT: --- | '

        if pose.right:
            pos = pose.right.position
            output += f'RIGHT: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})'
        else:
            output += 'RIGHT: ---'

        print(output, end='', flush=True)

    # Connect to real-time stream
    print('Streaming controller data... (Press Ctrl+C to stop)\n')

    try:
        await client.connect_stream(on_pose_update)
    except KeyboardInterrupt:
        print('\n\nStopped by user')
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())
