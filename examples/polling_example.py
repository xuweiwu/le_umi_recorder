"""
Polling Example - REST API Usage
Demonstrates using the REST API to poll for controller data
(Alternative to WebSocket streaming for simpler use cases)
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from api import ControllerTrackingClient


async def main():
    """Poll the REST API for controller data"""

    client = ControllerTrackingClient('http://localhost:8000')

    print('Controller Tracking - Polling Mode\n')

    try:
        # Get server status
        print('Server Status:')
        status = await client.get_status()
        for key, value in status.items():
            print(f'  {key}: {value}')
        print()

        # Poll for data at 10 Hz
        poll_rate = 10  # Hz
        poll_interval = 1.0 / poll_rate

        print(f'Polling at {poll_rate} Hz... (Press Ctrl+C to stop)\n')

        frame_count = 0

        while True:
            # Get latest pose
            pose = await client.get_latest_pose()

            if pose:
                frame_count += 1

                # Print controller data
                print(f'Frame {frame_count}:')

                if pose.left:
                    pos = pose.left.position
                    ori = pose.left.orientation
                    print(f'  Left Controller:')
                    print(f'    Position:    ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})')
                    print(f'    Orientation: ({ori[0]:7.3f}, {ori[1]:7.3f}, {ori[2]:7.3f}, {ori[3]:7.3f})')

                if pose.right:
                    pos = pose.right.position
                    ori = pose.right.orientation
                    print(f'  Right Controller:')
                    print(f'    Position:    ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})')
                    print(f'    Orientation: ({ori[0]:7.3f}, {ori[1]:7.3f}, {ori[2]:7.3f}, {ori[3]:7.3f})')

                print(f'  Timestamp: {pose.timestamp:.3f}')
                print(f'  Coord System: {pose.coordinate_system}')
                print()

            else:
                print('No pose data available')

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    except KeyboardInterrupt:
        print('\nStopped by user')
    finally:
        await client.close()


async def individual_controller_example():
    """Example: Query individual controllers"""

    client = ControllerTrackingClient('http://localhost:8000')

    try:
        # Get left controller only
        left = await client.get_left_controller()
        if left:
            print(f'Left Controller: {left.position}')

        # Get right controller only
        right = await client.get_right_controller()
        if right:
            print(f'Right Controller: {right.position}')

    finally:
        await client.close()


if __name__ == '__main__':
    # Run main polling example
    asyncio.run(main())

    # Or run individual controller example
    # asyncio.run(individual_controller_example())
