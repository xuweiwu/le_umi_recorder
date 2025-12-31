"""
LeRobot Dataset Visualizer
Uses LeRobot's built-in Rerun visualization when available.
Falls back to custom visualization for standalone use.

Usage with LeRobot installed:
    python -m umi.visualize_dataset --dataset ./datasets/my_dataset --episode 0

Usage without LeRobot (standalone):
    python -m umi.visualize_dataset --dataset ./datasets/my_dataset --episode 0 --standalone
"""

import argparse
import subprocess
import sys
from pathlib import Path


def use_lerobot_visualizer(dataset_path: str, episode_index: int = 0):
    """
    Use LeRobot's built-in visualize_dataset.py with Rerun.

    This is the recommended way to visualize LeRobot datasets.
    Requires: pip install lerobot

    See: https://github.com/huggingface/lerobot
    """
    # Convert to absolute path
    dataset_path = str(Path(dataset_path).resolve())

    # Extract repo-id style path (last two components or just the name)
    path_parts = Path(dataset_path).parts
    if len(path_parts) >= 2:
        repo_id = f"{path_parts[-2]}/{path_parts[-1]}"
    else:
        repo_id = path_parts[-1]

    # Get the parent directory as root
    root_dir = str(Path(dataset_path).parent.parent)

    print(f"Using LeRobot's built-in visualizer...")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Repo ID: {repo_id}")
    print(f"  Root: {root_dir}")
    print(f"  Episode: {episode_index}")
    print()

    # Try the CLI command first (lerobot >= 0.4.0)
    cmd = [
        sys.executable, "-m", "lerobot.scripts.visualize_dataset",
        "--repo-id", repo_id,
        "--root", root_dir,
        "--episode-index", str(episode_index),
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running LeRobot visualizer: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("LeRobot not found. Install with: pip install lerobot")
        print("Or use --standalone flag for custom visualization.")
        sys.exit(1)


def standalone_visualize(
    dataset_path: str,
    episode_index: int = 0,
    playback_speed: float = 1.0,
):
    """
    Standalone visualization without LeRobot dependency.
    Uses custom LeRobotFormatReader and Rerun directly.
    """
    import time
    import numpy as np

    try:
        import rerun as rr
    except ImportError:
        print("Error: rerun-sdk not installed. Install with: pip install rerun-sdk>=0.20.0")
        return

    try:
        from .lerobot_writer import LeRobotFormatReader
    except ImportError:
        from lerobot_writer import LeRobotFormatReader

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    reader = LeRobotFormatReader(dataset_path)

    print(f"Dataset info:")
    print(f"  FPS: {reader.fps}")
    print(f"  Episodes: {reader.num_episodes}")
    print(f"  Total frames: {reader.total_frames}")
    print(f"  Features: {list(reader.info.get('features', {}).keys())}")

    # Check episode exists
    if episode_index >= reader.num_episodes:
        print(f"Error: Episode {episode_index} not found. Dataset has {reader.num_episodes} episodes.")
        return

    episode = reader.episodes[episode_index]
    episode_length = episode.get('num_frames', 0)
    task = episode.get('task', 'Unknown')

    print(f"\nPlaying episode {episode_index}:")
    print(f"  Task: {task}")
    print(f"  Frames: {episode_length}")
    print(f"  Duration: {episode_length / reader.fps:.1f}s")

    # Load episode data
    data = reader.read_episode_as_dict(episode_index)

    # Load video frames if available
    video_features = reader.get_video_features()
    video_frames = {}
    for video_feat in video_features:
        try:
            video_frames[video_feat] = reader.read_video(episode_index, video_feat)
            print(f"  Loaded {len(video_frames[video_feat])} frames for {video_feat}")
        except FileNotFoundError:
            print(f"  Warning: Video not found for {video_feat}")

    # Initialize Rerun
    dataset_name = Path(dataset_path).name
    rr.init(f"umi_playback_{dataset_name}", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Calculate frame interval
    frame_interval = 1.0 / (reader.fps * playback_speed)

    # Playback loop
    print(f"\nPlaying at {playback_speed}x speed...")
    print("Press Ctrl+C to stop\n")

    try:
        for i in range(episode_length):
            loop_start = time.time()

            # Set timeline (using new Rerun 0.23+ API)
            rr.set_time("frame", sequence=i)
            rr.set_time("time", duration=i / reader.fps)

            # Log action (controller poses)
            if 'action' in data:
                action = data['action'][i]
                if isinstance(action, list):
                    action = np.array(action)

                # Parse left controller (first 7 values)
                if len(action) >= 7:
                    left_pos = action[0:3]
                    left_quat = action[3:7]  # [qx, qy, qz, qw]

                    rr.log(
                        'controllers/left',
                        rr.Transform3D(
                            translation=left_pos,
                            rotation=rr.Quaternion(xyzw=left_quat),
                            axis_length=0.15,
                        ),
                    )

                # Parse right controller (values 7-14)
                if len(action) >= 14:
                    right_pos = action[7:10]
                    right_quat = action[10:14]

                    rr.log(
                        'controllers/right',
                        rr.Transform3D(
                            translation=right_pos,
                            rotation=rr.Quaternion(xyzw=right_quat),
                            axis_length=0.15,
                        ),
                    )

            # Log observation state (gripper values)
            if 'observation.state' in data:
                state = data['observation.state'][i]
                if isinstance(state, list):
                    state = np.array(state)
                state_text = f"Gripper L: {state[0]:.2f}\nGripper R: {state[1]:.2f}" if len(state) >= 2 else str(state)
                rr.log('state', rr.TextDocument(state_text))

            # Log camera images
            for video_feat, frames in video_frames.items():
                if i < len(frames):
                    image = frames[i]
                    camera_name = video_feat.replace('observation.images.', '')
                    rr.log(f'cameras/{camera_name}', rr.Image(image))

            # Log playback status
            progress = (i + 1) / episode_length * 100
            status = f"## Playback\n\n"
            status += f"- **Episode**: {episode_index}\n"
            status += f"- **Task**: {task}\n"
            status += f"- **Frame**: {i + 1}/{episode_length}\n"
            status += f"- **Progress**: {progress:.1f}%\n"
            status += f"- **Time**: {i / reader.fps:.2f}s"
            rr.log('ui/status', rr.TextDocument(status, media_type=rr.MediaType.MARKDOWN))

            # Print progress
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Frame {i + 1}/{episode_length} ({progress:.1f}%)")

            # Maintain playback speed
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\nPlayback stopped by user")

    print(f"\nPlayback complete!")


def list_episodes(dataset_path: str):
    """List all episodes in a dataset."""
    try:
        from .lerobot_writer import LeRobotFormatReader
    except ImportError:
        from lerobot_writer import LeRobotFormatReader

    reader = LeRobotFormatReader(dataset_path)

    print(f"Dataset: {dataset_path}")
    print(f"FPS: {reader.fps}")
    print(f"Total episodes: {reader.num_episodes}")
    print(f"Total frames: {reader.total_frames}")
    print()

    for i, ep in enumerate(reader.episodes):
        num_frames = ep.get('num_frames', 0)
        duration = num_frames / reader.fps
        task = ep.get('task', 'Unknown')
        print(f"  Episode {i}: {num_frames} frames ({duration:.1f}s) - {task}")


def check_lerobot_available() -> bool:
    """Check if LeRobot is installed."""
    try:
        import lerobot
        return True
    except ImportError:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize LeRobot dataset in Rerun',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize using LeRobot's built-in visualizer (recommended)
  python -m umi.visualize_dataset --dataset ./datasets/2024_01_01/my_dataset --episode 0

  # Use standalone mode (no lerobot dependency)
  python -m umi.visualize_dataset --dataset ./datasets/my_dataset --episode 0 --standalone

  # Play at 2x speed
  python -m umi.visualize_dataset --dataset ./datasets/my_dataset --episode 2 --speed 2.0 --standalone

  # List all episodes
  python -m umi.visualize_dataset --dataset ./datasets/my_dataset --list

For more information on LeRobot visualization:
  https://github.com/huggingface/lerobot
  https://huggingface.co/docs/lerobot/lerobot-dataset-v3
        """
    )

    parser.add_argument('--dataset', required=True, help='Local path to dataset')
    parser.add_argument('--episode', type=int, default=0, help='Episode index to visualize (default: 0)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (default: 1.0, standalone only)')
    parser.add_argument('--list', action='store_true', help='List all episodes and exit')
    parser.add_argument('--standalone', action='store_true',
                        help='Use standalone visualization (no lerobot dependency)')

    args = parser.parse_args()

    if args.list:
        list_episodes(args.dataset)
        return

    # Check if LeRobot is available
    lerobot_available = check_lerobot_available()

    if args.standalone or not lerobot_available:
        if not args.standalone and not lerobot_available:
            print("LeRobot not installed. Using standalone visualization.")
            print("Install LeRobot for the full experience: pip install lerobot")
            print()

        standalone_visualize(
            dataset_path=args.dataset,
            episode_index=args.episode,
            playback_speed=args.speed,
        )
    else:
        use_lerobot_visualizer(
            dataset_path=args.dataset,
            episode_index=args.episode,
        )


if __name__ == '__main__':
    main()
