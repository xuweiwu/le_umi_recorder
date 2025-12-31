"""
Standalone LeRobot Format Writer
Writes datasets in LeRobot v2.0 compatible format without requiring the lerobot package
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class FeatureDefinition:
    """Definition of a dataset feature."""
    dtype: str  # "float32", "int64", "video", "image"
    shape: tuple
    names: Optional[Dict[str, List[str]]] = None


@dataclass
class EpisodeInfo:
    """Metadata for a single episode."""
    episode_index: int
    num_frames: int
    task: str
    length_s: float  # Duration in seconds


class LeRobotFormatWriter:
    """
    Standalone writer for LeRobot v2.0 compatible datasets.

    Creates datasets with:
    - Parquet files for tabular data (one per episode)
    - MP4 files for video observations (one per camera per episode)
    - JSON metadata files

    Usage:
        writer = LeRobotFormatWriter(
            output_dir="./datasets/my_dataset",
            fps=30,
            features={
                "action": FeatureDefinition("float32", (14,)),
                "observation.state": FeatureDefinition("float32", (2,)),
                "observation.images.camera_0": FeatureDefinition("video", (480, 640, 3)),
            }
        )

        # Record episode
        writer.start_episode()
        for frame in frames:
            writer.add_frame(frame)
        writer.end_episode(task="Pick up object")

        # Finalize
        writer.finalize()
    """

    def __init__(
        self,
        output_dir: str,
        fps: int = 30,
        features: Optional[Dict[str, FeatureDefinition]] = None,
        robot_type: str = "umi",
        repo_id: Optional[str] = None,
    ):
        """
        Initialize the writer.

        Args:
            output_dir: Output directory for the dataset
            fps: Frames per second
            features: Feature definitions
            robot_type: Robot type identifier
            repo_id: Optional repository ID
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.features = features or {}
        self.robot_type = robot_type
        self.repo_id = repo_id or output_dir

        # Create directory structure
        self.data_dir = self.output_dir / "data"
        self.videos_dir = self.output_dir / "videos"
        self.meta_dir = self.output_dir / "meta"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Episode state
        self.episodes: List[EpisodeInfo] = []
        self.current_episode_index = 0
        self.episode_buffer: List[Dict[str, Any]] = []
        self.video_buffers: Dict[str, List[np.ndarray]] = {}

        # Stats accumulator
        self._stats_accum: Dict[str, Dict[str, Any]] = {}

        # Identify video features
        self.video_features = [
            name for name, feat in self.features.items()
            if feat.dtype in ("video", "image")
        ]

        # Create video directories
        for video_feat in self.video_features:
            video_dir = self.videos_dir / video_feat
            video_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(self):
        """Start a new episode."""
        self.episode_buffer = []
        self.video_buffers = {name: [] for name in self.video_features}

    def add_frame(self, frame_data: Dict[str, Any]):
        """
        Add a frame to the current episode.

        Args:
            frame_data: Dictionary with feature names as keys
        """
        # Separate video and tabular data
        tabular_data = {}

        for key, value in frame_data.items():
            if key in self.video_features:
                # Buffer video frames
                if isinstance(value, np.ndarray):
                    self.video_buffers[key].append(value)
            else:
                # Store tabular data
                if isinstance(value, np.ndarray):
                    tabular_data[key] = value.tolist() if value.ndim > 0 else float(value)
                else:
                    tabular_data[key] = value

        # Add frame index and episode index
        tabular_data["frame_index"] = len(self.episode_buffer)
        tabular_data["episode_index"] = self.current_episode_index
        tabular_data["timestamp"] = len(self.episode_buffer) / self.fps

        self.episode_buffer.append(tabular_data)

    def end_episode(self, task: str = ""):
        """
        End the current episode and save to disk.

        Args:
            task: Task description for this episode
        """
        if not self.episode_buffer:
            print("Warning: No frames in episode, skipping save")
            return

        num_frames = len(self.episode_buffer)
        episode_idx = self.current_episode_index

        # Save parquet file
        self._save_parquet(episode_idx)

        # Save video files
        for video_feat in self.video_features:
            if self.video_buffers.get(video_feat):
                self._save_video(episode_idx, video_feat, self.video_buffers[video_feat])

        # Update stats
        self._update_stats()

        # Record episode info
        episode_info = EpisodeInfo(
            episode_index=episode_idx,
            num_frames=num_frames,
            task=task,
            length_s=num_frames / self.fps,
        )
        self.episodes.append(episode_info)

        # Prepare for next episode
        self.current_episode_index += 1
        self.episode_buffer = []
        self.video_buffers = {name: [] for name in self.video_features}

        print(f"Saved episode {episode_idx} with {num_frames} frames")

    def clear_episode_buffer(self):
        """Discard the current episode buffer without saving."""
        self.episode_buffer = []
        self.video_buffers = {name: [] for name in self.video_features}

    def _save_parquet(self, episode_idx: int):
        """Save episode data to parquet file."""
        if not self.episode_buffer:
            return

        # Convert to columnar format
        columns = {}
        for key in self.episode_buffer[0].keys():
            columns[key] = [frame.get(key) for frame in self.episode_buffer]

        # Create pyarrow table
        table = pa.table(columns)

        # Save to parquet
        filename = f"episode_{episode_idx:06d}.parquet"
        filepath = self.data_dir / filename
        pq.write_table(table, filepath)

    def _save_video(self, episode_idx: int, feature_name: str, frames: List[np.ndarray]):
        """Save video frames to MP4 file."""
        if not frames:
            return

        video_dir = self.videos_dir / feature_name
        video_dir.mkdir(parents=True, exist_ok=True)

        filename = f"episode_{episode_idx:06d}.mp4"
        filepath = video_dir / filename

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Try to use ffmpeg for better compression, fall back to OpenCV
        if self._has_ffmpeg():
            self._save_video_ffmpeg(filepath, frames, width, height)
        else:
            self._save_video_opencv(filepath, frames, width, height)

    def _has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _save_video_ffmpeg(self, filepath: Path, frames: List[np.ndarray], width: int, height: int):
        """Save video using ffmpeg (better compression)."""
        # Write frames to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                # Convert RGB to BGR for saving
                if frame.shape[-1] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                cv2.imwrite(f"{tmpdir}/frame_{i:06d}.png", frame_bgr)

            # Encode with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", f"{tmpdir}/frame_%06d.png",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                str(filepath)
            ]
            subprocess.run(cmd, capture_output=True, check=True)

    def _save_video_opencv(self, filepath: Path, frames: List[np.ndarray], width: int, height: int):
        """Save video using OpenCV (fallback)."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)

        writer.release()

    def _update_stats(self):
        """Update running statistics from episode buffer."""
        for frame in self.episode_buffer:
            for key, value in frame.items():
                if key in ("frame_index", "episode_index", "timestamp"):
                    continue

                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                elif not isinstance(value, np.ndarray):
                    value = np.array([value])

                if key not in self._stats_accum:
                    self._stats_accum[key] = {
                        "sum": np.zeros_like(value, dtype=np.float64),
                        "sum_sq": np.zeros_like(value, dtype=np.float64),
                        "min": np.full_like(value, np.inf, dtype=np.float64),
                        "max": np.full_like(value, -np.inf, dtype=np.float64),
                        "count": 0,
                    }

                stats = self._stats_accum[key]
                stats["sum"] += value
                stats["sum_sq"] += value ** 2
                stats["min"] = np.minimum(stats["min"], value)
                stats["max"] = np.maximum(stats["max"], value)
                stats["count"] += 1

    def _compute_stats(self) -> Dict[str, Dict[str, List[float]]]:
        """Compute final statistics."""
        stats = {}
        for key, accum in self._stats_accum.items():
            count = accum["count"]
            if count == 0:
                continue

            mean = accum["sum"] / count
            variance = (accum["sum_sq"] / count) - (mean ** 2)
            std = np.sqrt(np.maximum(variance, 0))

            stats[key] = {
                "mean": mean.tolist() if hasattr(mean, 'tolist') else [float(mean)],
                "std": std.tolist() if hasattr(std, 'tolist') else [float(std)],
                "min": accum["min"].tolist() if hasattr(accum["min"], 'tolist') else [float(accum["min"])],
                "max": accum["max"].tolist() if hasattr(accum["max"], 'tolist') else [float(accum["max"])],
            }

        return stats

    def finalize(self):
        """Finalize the dataset and write metadata files."""
        # Write info.json
        self._write_info_json()

        # Write stats.json
        self._write_stats_json()

        # Write episodes.jsonl
        self._write_episodes_jsonl()

        print(f"\nDataset finalized at: {self.output_dir}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Total frames: {sum(ep.num_frames for ep in self.episodes)}")

    def _write_info_json(self):
        """Write dataset info metadata."""
        # Build features dict for JSON
        features_json = {}
        for name, feat in self.features.items():
            features_json[name] = {
                "dtype": feat.dtype,
                "shape": list(feat.shape),
            }
            if feat.names:
                features_json[name]["names"] = feat.names

        info = {
            "codebase_version": "v2.0",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": features_json,
            "total_episodes": len(self.episodes),
            "total_frames": sum(ep.num_frames for ep in self.episodes),
            "repo_id": self.repo_id,
        }

        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _write_stats_json(self):
        """Write normalization statistics."""
        stats = self._compute_stats()

        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def _write_episodes_jsonl(self):
        """Write episode metadata as JSONL."""
        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for ep in self.episodes:
                ep_dict = {
                    "episode_index": ep.episode_index,
                    "num_frames": ep.num_frames,
                    "task": ep.task,
                    "length_s": ep.length_s,
                }
                f.write(json.dumps(ep_dict) + "\n")


class LeRobotFormatReader:
    """
    Reader for LeRobot format datasets.

    Usage:
        reader = LeRobotFormatReader("./datasets/my_dataset")
        print(reader.info)
        print(reader.episodes)

        # Read episode data
        data = reader.read_episode(0)
        frames = reader.read_video(0, "observation.images.camera_0")
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir / "data"
        self.videos_dir = self.dataset_dir / "videos"
        self.meta_dir = self.dataset_dir / "meta"

        # Load metadata
        self.info = self._load_info()
        self.stats = self._load_stats()
        self.episodes = self._load_episodes()

    def _load_info(self) -> Dict:
        """Load info.json."""
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
        return {}

    def _load_stats(self) -> Dict:
        """Load stats.json."""
        stats_path = self.meta_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                return json.load(f)
        return {}

    def _load_episodes(self) -> List[Dict]:
        """Load episodes.jsonl."""
        episodes = []
        episodes_path = self.meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path) as f:
                for line in f:
                    if line.strip():
                        episodes.append(json.loads(line))
        return episodes

    @property
    def fps(self) -> int:
        return self.info.get("fps", 30)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(ep.get("num_frames", 0) for ep in self.episodes)

    def read_episode(self, episode_index: int) -> pa.Table:
        """Read episode data from parquet file."""
        filename = f"episode_{episode_index:06d}.parquet"
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Episode {episode_index} not found: {filepath}")
        return pq.read_table(filepath)

    def read_episode_as_dict(self, episode_index: int) -> Dict[str, List]:
        """Read episode data as dictionary of lists."""
        table = self.read_episode(episode_index)
        return {col: table[col].to_pylist() for col in table.column_names}

    def read_video(self, episode_index: int, feature_name: str) -> List[np.ndarray]:
        """Read video frames for an episode."""
        filename = f"episode_{episode_index:06d}.mp4"
        filepath = self.videos_dir / feature_name / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Video not found: {filepath}")

        frames = []
        cap = cv2.VideoCapture(str(filepath))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    def get_video_features(self) -> List[str]:
        """Get list of video feature names."""
        return [
            name for name, feat in self.info.get("features", {}).items()
            if feat.get("dtype") in ("video", "image")
        ]


def main():
    """Test the writer and reader."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test dataset
        features = {
            "action": FeatureDefinition("float32", (14,)),
            "observation.state": FeatureDefinition("float32", (2,)),
        }

        writer = LeRobotFormatWriter(
            output_dir=tmpdir,
            fps=30,
            features=features,
            robot_type="test",
        )

        # Record 2 episodes
        for ep in range(2):
            writer.start_episode()
            for i in range(10):
                writer.add_frame({
                    "action": np.random.randn(14).astype(np.float32),
                    "observation.state": np.random.randn(2).astype(np.float32),
                })
            writer.end_episode(task=f"Test task {ep}")

        writer.finalize()

        # Read it back
        reader = LeRobotFormatReader(tmpdir)
        print(f"\nDataset info: {reader.info}")
        print(f"Episodes: {reader.num_episodes}")
        print(f"Total frames: {reader.total_frames}")

        data = reader.read_episode_as_dict(0)
        print(f"\nEpisode 0 keys: {list(data.keys())}")
        print(f"Episode 0 frames: {len(data['action'])}")


if __name__ == "__main__":
    main()
