# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os

from device_specs import DeviceClass, get_board_spec, get_deployment


@dataclass
class WanConfig:
    """Configuration for Wan2.2 T2V standalone server.

    Topology fields (num_workers, device_mesh_shape, device_ids, fabric) are
    derived from device_specs.DEPLOYMENTS based on `board`.
    """

    # Required: which Tenstorrent board this server is running on.
    board: DeviceClass = None  # set explicitly via CLI; validated in __post_init__

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # Topology — derived from device_specs.get_deployment("wan22", board) in __post_init__.
    num_workers: int = 1
    device_ids: Tuple[int, ...] = ()
    device_mesh_shape: Tuple[int, int] = (0, 0)
    fabric_config_name: str = "FABRIC_1D"

    # Device parameters. l1_small_size is filled from BoardSpec; trace_region_size
    # is per-model (Wan2.2 ships at 30MB per tt-inference-server/workflows/model_spec.py).
    l1_small_size: int = 0
    trace_region_size: int = 30_000_000

    # Model settings
    model_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    hf_home: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Pipeline / generation defaults
    num_inference_steps: int = 40
    num_frames: int = 81
    height: int = 480
    width: int = 832
    guidance_scale: float = 4.0
    guidance_scale_2: float = 3.0
    use_trace: bool = True

    # Queue settings
    max_queue_size: int = 4
    inference_timeout_seconds: int = 1800  # video gen is slow

    # Development mode: WAN_DEV_MODE=true → fewer steps, fewer frames, no trace.
    dev_mode: bool = field(default_factory=lambda: os.getenv("WAN_DEV_MODE", "false").lower() == "true")

    def __post_init__(self):
        if self.board is None:
            raise ValueError("WanConfig.board is required (pass --board on the CLI)")

        dep = get_deployment("wan22", self.board)
        self.num_workers = dep.num_workers
        self.device_mesh_shape = dep.mesh_shape
        self.device_ids = dep.device_ids
        if dep.fabric_config:
            self.fabric_config_name = dep.fabric_config

        spec = get_board_spec(self.board)
        if self.l1_small_size == 0:
            self.l1_small_size = spec.l1_small_size

        if self.dev_mode:
            self.num_inference_steps = 4
            self.use_trace = False
            # NOTE: num_frames is fixed at pipeline creation: WanPipeline.__init__
            # internally calls warmup_buffers(num_frames=81), which allocates
            # self.latent_buffer at the 81-frame latent size. Subsequent calls
            # with a different num_frames break ttnn.copy(latent_buffer). So we
            # leave num_frames=81 even in dev mode.
