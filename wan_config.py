# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class WANConfig:
    """Configuration for WAN 2.2 T2V standalone server (2x4 mesh / LoudBox)"""

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    num_workers: int = 1  # Single worker owns entire 2x4 mesh

    # Device settings (LoudBox) - 1 worker with 2x4 mesh
    device_ids: Tuple[int, ...] = (
        0,
        1,
        2,
        3,
    )  # PCIe-attached L chips only; R chips auto-discovered via ethernet fabric
    device_mesh_shape: Tuple[int, int] = (2, 4)

    # Device parameters (matching SD3.5 / WAN pipeline requirements)
    fabric_config_name: str = "FABRIC_1D"  # String (not enum) for pickle compatibility
    l1_small_size: int = 32768
    trace_region_size: int = 25000000

    # Model settings
    model_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # Pipeline settings
    num_inference_steps: int = 20
    guidance_scale: float = 3.0
    guidance_scale_2: float = 4.0
    video_width: int = 832
    video_height: int = 480
    num_frames: int = 81  # ~5s at 16fps
    fps: int = 16

    # Queue settings
    max_queue_size: int = 4  # Smaller than image (video generation is slow)
    inference_timeout_seconds: int = 1200  # 20 min — inference observed at ~364s on 81 frames

    # Cache directory for TT DiT compiled weights; reads TT_DIT_CACHE_DIR env var
    # Falls back to ~/.cache/tt-dit so weights are always loaded from disk, not
    # re-fragmented into DRAM from raw PyTorch state dicts on every warmup step.
    tt_dit_cache_dir: str = field(
        default_factory=lambda: os.getenv("TT_DIT_CACHE_DIR", os.path.expanduser("~/.cache/tt-dit"))
    )

    # Development mode: reads WAN_DEV_MODE env var
    # When True: num_inference_steps=5 (faster dev iteration)
    dev_mode: bool = field(default_factory=lambda: os.getenv("WAN_DEV_MODE", "false").lower() == "true")

    # Skip warmup inference: reads WAN_SKIP_WARMUP env var
    # When True: skips the warmup run on startup (useful for one-off runs)
    skip_warmup: bool = field(default_factory=lambda: os.getenv("WAN_SKIP_WARMUP", "false").lower() == "true")

    def __post_init__(self):
        """Apply dev mode overrides if enabled"""
        if self.dev_mode:
            self.num_inference_steps = 5  # Fewer steps for faster dev iteration
