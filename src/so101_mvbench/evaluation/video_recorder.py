"""Async MP4 video recorder for evaluation episodes.

Captures camera frames at a fixed interval and encodes them to MP4
in a background thread (ThreadPoolExecutor). Frames are resized before
encoding to reduce file size.
"""

VERSION = "1.1.0"

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from so101_mvbench.logging import get_logger

logger = get_logger(__name__)

# Default settings — documented in CSV/JSON output
DEFAULT_FPS = 10
DEFAULT_INTERVAL = 15       # every 15 steps = 0.25s sim-time per frame
DEFAULT_RESIZE = (320, 240)
DEFAULT_SPEEDUP = 2.5       # 10fps × 0.25s/frame = 2.5× real-time


class VideoRecorder:
    """Async MP4 recorder for camera streams the policy consumes.

    By default (``policy_cams=None``) captures both ``camera_ego`` and
    ``camera_external`` — for backwards compatibility with debug workflows.
    Passing ``policy_cams=list(rename_map.keys())`` filters to only the
    cameras the policy actually consumes, so 1-cam policies don't emit a
    misleading ``external.mp4`` that was never fed to the model.
    """

    def __init__(
        self,
        base_dir: Path | None,
        policy_cams: list[str] | None = None,
        fps: int = DEFAULT_FPS,
        interval: int = DEFAULT_INTERVAL,
        resize: tuple[int, int] = DEFAULT_RESIZE,
    ):
        self._base_dir = base_dir
        self._policy_cams = set(policy_cams) if policy_cams else None
        self._fps = fps
        self._interval = interval
        self._resize = resize
        self._frame_buffers: dict[str, list[np.ndarray]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures = []

    @property
    def settings(self) -> dict:
        """Video settings dict for JSON summary."""
        return {
            "format": "mp4",
            "fps": self._fps,
            "capture_interval_steps": self._interval,
            "sim_time_per_frame_s": self._interval / 60,  # 60Hz effective rate
            "playback_speedup": DEFAULT_SPEEDUP,
            "resolution": f"{self._resize[0]}x{self._resize[1]}",
            "policy_cams": sorted(self._policy_cams) if self._policy_cams else None,
        }

    @property
    def enabled(self) -> bool:
        return self._base_dir is not None

    def capture(self, obs: dict, step: int) -> None:
        """Grab camera frames if step matches interval.

        Cameras not in ``self._policy_cams`` are skipped — avoids emitting
        MP4s for cameras the policy never consumed.
        """
        if not self.enabled or step % self._interval != 0:
            return
        for cam_key, img_tensor in obs.get("visual", {}).items():
            cam_name = cam_key.replace("camera_", "", 1)
            if self._policy_cams is not None and cam_name not in self._policy_cams:
                continue
            if cam_name not in self._frame_buffers:
                self._frame_buffers[cam_name] = []
            img = img_tensor[0].cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            self._frame_buffers[cam_name].append(img)

    def flush(self, ep_dir: Path) -> None:
        """Encode collected frames to MP4 async, then clear buffers."""
        if not any(self._frame_buffers.values()):
            return
        ep_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {name: list(frames) for name, frames in self._frame_buffers.items()}
        self._futures.append(self._executor.submit(
            _encode_videos, snapshot, ep_dir, self._resize, self._fps,
        ))
        for frames in self._frame_buffers.values():
            frames.clear()

    def wait(self) -> None:
        """Wait for all pending encoding jobs."""
        if self._futures:
            logger.info("Waiting for %d video encoding jobs...", len(self._futures))
            for fut in self._futures:
                fut.result()
            self._executor.shutdown(wait=False)
            logger.info("Video encoding complete")


def _encode_videos(
    frame_buffers: dict[str, list[np.ndarray]],
    out_dir: Path,
    resize: tuple[int, int],
    fps: int,
) -> None:
    """Encode camera frames to MP4 (runs in background thread)."""
    import imageio
    from PIL import Image as _PILImg

    for cam_name, flist in frame_buffers.items():
        if flist:
            vid_path = out_dir / f"{cam_name}.mp4"
            writer = imageio.get_writer(str(vid_path), fps=fps)
            for f in flist:
                small = np.array(_PILImg.fromarray(f).resize(resize, _PILImg.LANCZOS))
                writer.append_data(small)
            writer.close()
