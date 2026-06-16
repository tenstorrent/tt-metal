# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import ttnn
import numpy as np
from typing import List

from wan_config import WanConfig
from utils.logger import setup_logger


class WanRunner:
    """Wrapper for tt-metal WanPipeline (Wan2.2 T2V).

    Mirrors SD35Runner in shape: initialize_device → load_model → run_inference → close_device.
    Returns numpy uint8 frames of shape (T, H, W, 3) per request.
    """

    def __init__(self, worker_id: int, config: WanConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"WanRunner-{worker_id}")
        self.mesh_device = None
        self.pipeline = None
        self._fabric_config = None

    def initialize_device(self):
        rows, cols = self.config.device_mesh_shape
        self.logger.info(
            f"Initializing {rows}x{cols} mesh device for worker {self.worker_id} "
            f"on board {self.config.board.name.lower()}"
        )

        fabric_config = getattr(ttnn.FabricConfig, self.config.fabric_config_name)
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
        from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

        self.logger.info("Creating Wan2.2 T2V pipeline...")
        self.logger.info(f"  model_name: {self.config.model_name}")
        self.logger.info(f"  size: {self.config.width}x{self.config.height}, frames: {self.config.num_frames}")
        self.logger.info(f"  steps: {self.config.num_inference_steps}, use_trace: {self.config.use_trace}")

        # Geometry (height/width/num_frames) and CFG are fixed at creation; guidance_scale
        # is applied per call. Enable CFG so per-request guidance_scale > 1 is accepted.
        self.pipeline = WanPipeline.create_pipeline(
            mesh_device=self.mesh_device,
            checkpoint_name=self.config.model_name,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            cfg_enabled=self.config.guidance_scale > 1,
        )
        self.logger.info("Wan2.2 pipeline created")

        self.logger.info("Running warmup inference (compiles kernels / captures trace)...")
        # num_frames/height/width are fixed at creation; __call__ does not accept them.
        self.pipeline(
            prompts=["A golden sunrise over mountain peaks"],
            num_inference_steps=2,
            guidance_scale=self.config.guidance_scale,
            guidance_scale_2=self.config.guidance_scale_2,
            seed=42,
            output_type="uint8",
            traced=self.config.use_trace,
        )
        self.logger.info("Warmup complete")

        if kernel_ready_queue is not None:
            kernel_ready_queue.put(self.worker_id)

    def run_inference(self, requests: List[dict]) -> List[np.ndarray]:
        request = requests[0]

        prompt = request["prompt"]
        negative_prompt = request.get("negative_prompt") or None
        num_inference_steps = request.get("num_inference_steps") or self.config.num_inference_steps
        # num_frames / height / width are fixed at pipeline creation; __call__ reads
        # self._num_frames/_height/_width and does not accept per-request overrides.
        guidance_scale = request.get("guidance_scale") or self.config.guidance_scale
        guidance_scale_2 = request.get("guidance_scale_2") or self.config.guidance_scale_2
        # flow_shift / boundary_ratio are host-side schedule/expert-selection knobs; pass
        # through only when supplied so the pipeline keeps its construction-time defaults.
        flow_shift = request.get("flow_shift")
        boundary_ratio = request.get("boundary_ratio")
        seed = request.get("seed")

        self.logger.info(
            f"Running inference: prompt='{prompt[:80]}', steps={num_inference_steps}, "
            f"frames={self.config.num_frames}, size={self.config.width}x{self.config.height}, seed={seed}"
        )

        kwargs = dict(
            prompts=[prompt],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            seed=int(seed) if seed is not None else 0,
            output_type="uint8",
            traced=self.config.use_trace,
        )
        if negative_prompt:
            kwargs["negative_prompts"] = [negative_prompt]
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        if boundary_ratio is not None:
            kwargs["boundary_ratio"] = float(boundary_ratio)

        output = self.pipeline(**kwargs)
        # WanPipelineOutput.frames: numpy uint8 of shape (B, T, H, W, C). Strip batch.
        frames = output.frames if hasattr(output, "frames") else output
        if frames.ndim == 5:
            frames = frames[0]
        return [frames]

    def close_device(self):
        if self.pipeline is not None and hasattr(self.pipeline, "synchronize_devices"):
            self.logger.info("Synchronizing submesh devices before closure")
            try:
                self.pipeline.synchronize_devices()
            except Exception as e:
                self.logger.warning(f"synchronize_devices failed: {e}")

        if self.mesh_device is not None:
            self.logger.info("Closing mesh device")
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None
            if self._fabric_config is not None:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                self._fabric_config = None
