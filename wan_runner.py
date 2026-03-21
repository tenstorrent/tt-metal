# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import ttnn
from typing import List

from wan_config import WANConfig
from utils.logger import setup_logger


class WANRunner:
    """
    Wrapper for WanPipeline that handles device initialization,
    model loading, warmup, and inference execution.

    WAN 2.2 T2V runs on a 2x4 mesh (LoudBox). The pipeline is called directly
    with prompt, height, width, num_frames, guidance_scale, guidance_scale_2,
    and seed. Output is output.frames — a numpy array of shape [T, H, W, C].
    """

    def __init__(self, worker_id: int, config: WANConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"WANRunner-{worker_id}")
        self.mesh_device = None
        self.pipeline = None
        self._fabric_config = None

    def initialize_device(self):
        """Initialize 2x4 mesh device with FABRIC_1D and WAN-specific parameters.

        On a LoudBox, 4 n300 boards each have a PCIe-attached L chip and a
        remote R chip connected via ethernet, giving 8 total devices (IDs 0-7).
        Only the L chips appear in /dev/tenstorrent; the R chips are discovered
        by ttnn via the fabric when opening the mesh.
        """
        self.logger.info(
            f"Initializing {self.config.device_mesh_shape[0]}x{self.config.device_mesh_shape[1]} "
            f"mesh device for worker {self.worker_id}"
        )

        rows, cols = self.config.device_mesh_shape

        # Map the fabric_config_name string to the ttnn enum value
        fabric_config = getattr(ttnn.FabricConfig, self.config.fabric_config_name)

        # Set fabric before opening the mesh (correct two-step API)
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
        )
        self._fabric_config = fabric_config

        self.mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            l1_small_size=self.config.l1_small_size,
            trace_region_size=self.config.trace_region_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(),
        )

        self.logger.info(
            f"Mesh device initialized: shape={tuple(self.mesh_device.shape)}, "
            f"fabric={self.config.fabric_config_name}, "
            f"l1_small_size={self.config.l1_small_size}, "
            f"trace_region_size={self.config.trace_region_size}"
        )
        return self.mesh_device

    def load_model(self, kernel_ready_queue=None):
        """Create WAN pipeline and run warmup inference.

        The pipeline is created via WanPipeline.create_pipeline() which handles
        all model loading and device setup. A single warmup run is performed to
        compile kernels.

        Args:
            kernel_ready_queue: Optional queue to signal when the pipeline is ready
                                for inference (warmup complete). For WAN with
                                num_workers=1, the server never waits on this queue,
                                but the signal is sent for protocol consistency.
        """
        # Deferred import: avoid importing ttnn/pipeline in the main process
        from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

        self.logger.info("Creating WAN 2.2 T2V pipeline...")
        self.logger.info(f"  model_name: {self.config.model_name}")
        self.logger.info(f"  video size: {self.config.video_width}x{self.config.video_height}")
        self.logger.info(f"  num_frames: {self.config.num_frames}")
        self.logger.info(f"  guidance_scale: {self.config.guidance_scale}")
        self.logger.info(f"  guidance_scale_2: {self.config.guidance_scale_2}")
        self.logger.info(f"  num_inference_steps: {self.config.num_inference_steps}")

        self.pipeline = WanPipeline.create_pipeline(
            mesh_device=self.mesh_device,
            checkpoint_name=self.config.model_name,
        )
        self.logger.info("WAN pipeline created")

        # Warmup inference: compiles kernels.
        if not self.config.skip_warmup:
            self.logger.info("Running warmup inference (compiles kernels)...")
            self.pipeline(
                prompt="A golden sunrise over mountain peaks",
                negative_prompt="",
                height=self.config.video_height,
                width=self.config.video_width,
                num_frames=self.config.num_frames,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                guidance_scale_2=self.config.guidance_scale_2,
                seed=42,
            )
            self.logger.info("Warmup complete")
        else:
            self.logger.info("Skipping warmup inference (--no-warmup mode; kernels will compile on first request)")

        # Signal kernel ready — for WAN num_workers=1 so the server's overlapped
        # startup loop (range(num_workers - 1) == range(0)) never waits here, but
        # we send the signal for protocol consistency.
        if kernel_ready_queue is not None:
            self.logger.info(f"Worker {self.worker_id} signaling kernel ready")
            kernel_ready_queue.put(self.worker_id)

    def run_inference(self, requests: List[dict]):
        """Run inference for a single WAN T2V request.

        WAN processes one request at a time (single worker across 2x4 mesh).
        Only the first request in the list is used.

        Args:
            requests: List of request dicts. Only requests[0] is used.

        Returns:
            numpy array of video frames with shape [T, H, W, C].
        """
        request = requests[0]

        prompt = request["prompt"]
        negative_prompt = request.get("negative_prompt", "")
        num_inference_steps = request.get("num_inference_steps") or self.config.num_inference_steps
        seed = request.get("seed")

        self.logger.info(f"Running inference: prompt='{prompt[:80]}', steps={num_inference_steps}, seed={seed}")

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=self.config.video_height,
            width=self.config.video_width,
            num_frames=self.config.num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            guidance_scale_2=self.config.guidance_scale_2,
            seed=seed,
        )

        return output.frames

    def close_device(self):
        """Synchronize all submesh devices then close the mesh device"""
        if self.pipeline is not None:
            self.logger.info("Synchronizing submesh devices before closure")
            self.pipeline.synchronize_devices()

        if self.mesh_device is not None:
            self.logger.info("Closing mesh device")
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None
            if self._fabric_config is not None:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                self._fabric_config = None
