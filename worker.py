# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import time
import queue
import multiprocessing as mp

from utils.logger import setup_logger


def setup_sdxl_worker_environment(worker_id: int, config):
    """
    Set worker-specific environment variables for SDXL (T3K, one device per worker).

    Copied from sdxl_worker.py's setup_worker_environment() to keep worker.py
    self-contained without importing the legacy sdxl_worker module.
    """
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()

    # Set device visibility EARLY (before any ttnn operations)
    # This ensures the correct devices are visible when ttnn initializes
    device_ids_str = ",".join(map(str, config.device_ids))
    os.environ["TT_VISIBLE_DEVICES"] = device_ids_str
    os.environ["TT_METAL_VISIBLE_DEVICES"] = device_ids_str

    # Performance settings (Galaxy mode only)
    if config.is_galaxy:
        os.environ["TT_MM_THROTTLE_PERF"] = "5"

        tt_metal_home = os.environ.get("TT_METAL_HOME", os.getcwd())
        if config.device_mesh_shape == (1, 1):
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/" "n150_mesh_graph_descriptor.textproto"
            )
        elif config.device_mesh_shape == (2, 1):
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/" "n300_mesh_graph_descriptor.textproto"
            )


def setup_sd35_worker_environment(worker_id: int, config):
    """
    Set worker-specific environment variables for SD3.5 (LoudBox, single worker
    owns the full 2x4 mesh).

    All 8 device IDs are made visible so the mesh device can be opened with
    the full 2x4 topology. No per-device isolation needed.
    """
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()

    # SD3.5 needs the full 2x4 mesh (all 8 chips: 4 PCIe L + 4 ethernet R).
    # Do NOT restrict TT_VISIBLE_DEVICES — let UMD discover the full topology.
    # The shell may have set TT_VISIBLE_DEVICES to PCIe-only IDs; clear it here
    # so the system mesh auto-discovers all 8 chips correctly.
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)


def setup_wan_worker_environment(worker_id: int, config):
    """
    Set worker-specific environment variables for WAN 2.2 (LoudBox, single worker
    owns the full 2x4 mesh).

    All 8 device IDs are made visible so the mesh device can be opened with
    the full 2x4 topology. No per-device isolation needed.
    """
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()

    # Ensure the DiT weight cache is set before WANRunner initializes so that
    # _cache_root() in models/tt_dit/utils/cache.py returns a valid path and
    # weights are loaded deterministically from disk (prevents DRAM OOM caused
    # by fragmentation when weights are loaded from raw PyTorch state dicts).
    os.environ["TT_DIT_CACHE_DIR"] = config.tt_dit_cache_dir

    # WAN needs the full 2x4 mesh (all 8 chips: 4 PCIe L + 4 ethernet R).
    # Do NOT restrict TT_VISIBLE_DEVICES — let UMD discover the full topology.
    # The shell may have set TT_VISIBLE_DEVICES to PCIe-only IDs; clear it here
    # so the system mesh auto-discovers all 8 chips correctly.
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)


def device_worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    warmup_signal_queue: mp.Queue,
    kernel_ready_queue: mp.Queue,
    error_queue: mp.Queue,
    config,
):
    """
    Main worker process function that handles inference tasks.

    Supports SDXL, SD3.5, and WAN 2.2 via config type dispatch:
    - WANConfig   → WANRunner   (2x4 mesh, single worker, video output)
    - SD35Config  → SD35Runner  (2x4 mesh, single worker)
    - SDXLConfig  → SDXLRunner  (1x1 mesh, one worker per device)

    All runner/ttnn imports are deferred to this function so the main server
    process never initializes ttnn (which would conflict with the child
    processes' device ownership).

    Args:
        worker_id: Worker process ID
        task_queue: Queue for incoming tasks — each item is (task_id, request_dict)
        result_queue: Queue for results — each item is a dict with task_id/images/inference_time
        warmup_signal_queue: Queue to signal warmup completion to the server
        kernel_ready_queue: Queue to signal kernel compilation complete (overlapped startup)
        error_queue: Queue for error reporting to the server
        config: SDXLConfig or SD35Config instance
    """
    logger = setup_logger(f"Worker-{worker_id}")
    logger.info(f"Worker {worker_id} starting...")

    try:
        # Determine model type and set up environment + runner
        # Imports are deferred here to avoid ttnn initialization in the main process
        from wan_config import WANConfig
        from sd35_config import SD35Config

        if isinstance(config, WANConfig):
            setup_wan_worker_environment(worker_id, config)
            from wan_runner import WANRunner

            runner = WANRunner(worker_id, config)
        elif isinstance(config, SD35Config):
            setup_sd35_worker_environment(worker_id, config)
            from sd35_runner import SD35Runner

            runner = SD35Runner(worker_id, config)
        else:
            setup_sdxl_worker_environment(worker_id, config)
            from sdxl_runner import SDXLRunner

            runner = SDXLRunner(worker_id, config)

        # Initialize device and load model with optional overlapped startup signal
        runner.initialize_device()
        runner.load_model(kernel_ready_queue=kernel_ready_queue)

        # Signal to server that this worker has completed full warmup
        warmup_signal_queue.put(worker_id)
        logger.info(f"Worker {worker_id} ready for inference")

        # Main processing loop
        while True:
            try:
                # Block with timeout so we can detect shutdown cleanly
                task = task_queue.get(timeout=1)

                if task is None:  # Shutdown signal
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break

                task_id, request = task
                logger.info(f"Worker {worker_id} processing task {task_id}")

                # Run inference and measure wall-clock time
                start_time = time.time()
                images = runner.run_inference([request])
                inference_time = time.time() - start_time

                # Return result to server
                result_queue.put({"task_id": task_id, "images": images, "inference_time": inference_time})

                logger.info(f"Task {task_id} completed in {inference_time:.2f}s")

            except queue.Empty:
                # Normal timeout — no task available, loop again
                continue
            except Exception as e:
                import traceback

                logger.error(f"Error processing task: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                error_queue.put({"worker_id": worker_id, "error": str(e)})

        # Cleanup
        runner.close_device()
        logger.info(f"Worker {worker_id} shutdown complete")

    except Exception as e:
        import traceback

        logger.error(f"Worker {worker_id} fatal error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_queue.put({"worker_id": worker_id, "error": f"Fatal: {str(e)}"})
