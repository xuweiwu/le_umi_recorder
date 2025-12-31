"""
CSV Logger for Controller Tracking
Logs controller poses to a CSV file for later analysis
"""

import asyncio
import sys
import csv
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from api import ControllerTrackingClient


class CSVLogger:
    """Log controller tracking data to CSV"""

    def __init__(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tracking_data_{timestamp}.csv'

        self.filename = filename
        self.file = None
        self.writer = None
        self.frame_count = 0

    def open(self):
        """Open CSV file and write header"""
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        # Write header
        self.writer.writerow([
            'frame',
            'timestamp',
            'coordinate_system',
            'hand',
            'pos_x', 'pos_y', 'pos_z',
            'ori_x', 'ori_y', 'ori_z', 'ori_w',
            'button_0_pressed', 'button_1_pressed',
            'received_at'
        ])

        print(f'Logging to: {self.filename}')

    def log_pose(self, pose):
        """Log a pose frame to CSV"""

        self.frame_count += 1

        # Log each controller
        for hand in ['left', 'right']:
            controller = getattr(pose, hand)

            if controller:
                # Extract data
                pos = controller.position
                ori = controller.orientation

                # Get button states (first two buttons)
                btn0 = controller.buttons.get('0', {}).get('pressed', False)
                btn1 = controller.buttons.get('1', {}).get('pressed', False)

                # Write row
                self.writer.writerow([
                    self.frame_count,
                    pose.timestamp,
                    pose.coordinate_system,
                    hand,
                    pos[0], pos[1], pos[2],
                    ori[0], ori[1], ori[2], ori[3],
                    1 if btn0 else 0,
                    1 if btn1 else 0,
                    pose.received_at
                ])

        # Print progress
        if self.frame_count % 100 == 0:
            print(f'Logged {self.frame_count} frames...')

    def close(self):
        """Close the CSV file"""
        if self.file:
            self.file.close()
            print(f'\nClosed log file: {self.filename}')
            print(f'Total frames logged: {self.frame_count}')


async def main():
    """Main logging loop"""

    import argparse

    parser = argparse.ArgumentParser(description='Log controller tracking data to CSV')
    parser.add_argument(
        '--server',
        default='http://localhost:8000',
        help='Server URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--output',
        help='Output CSV filename (default: tracking_data_TIMESTAMP.csv)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Recording duration in seconds (default: unlimited)'
    )

    args = parser.parse_args()

    # Setup logger
    logger = CSVLogger(args.output)
    logger.open()

    # Setup client
    client = ControllerTrackingClient(args.server)

    print(f'Connecting to {args.server}...')

    # Check server
    try:
        status = await client.get_status()
        print(f'Server connected ({status["current_frame_rate"]} Hz)\n')
    except Exception as e:
        print(f'Error: {e}')
        return

    # Define callback
    def on_pose_update(pose):
        logger.log_pose(pose)

    # Start logging
    if args.duration:
        print(f'Recording for {args.duration} seconds...\n')
    else:
        print('Recording... (Press Ctrl+C to stop)\n')

    try:
        if args.duration:
            # Record for specified duration
            task = asyncio.create_task(client.connect_stream(on_pose_update))
            await asyncio.sleep(args.duration)
            task.cancel()
        else:
            # Record until interrupted
            await client.connect_stream(on_pose_update)

    except KeyboardInterrupt:
        print('\n\nStopped by user')
    except asyncio.CancelledError:
        print('\n\nRecording complete')
    finally:
        await client.close()
        logger.close()


if __name__ == '__main__':
    asyncio.run(main())
