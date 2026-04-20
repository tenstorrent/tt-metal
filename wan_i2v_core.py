"""Reusable Wan 2.2 I2V building blocks for CLI and HTTP server.

Split out of wan_i2v_generate.py so both the CLI and server/server.py can
share one implementation of:

  - CONFIGS, RESOLUTIONS, DEFAULT_NEGATIVE_PROMPT       (constants)
  - tt_device_lock(path, timeout)                        (fcntl flock CM)
  - open_mesh(config_name, use_lock)                     (mesh + fabric CM)
  - create_pipeline(mesh_device, cfg, h, w, num_frames)  (pipeline build)
  - generate_video(pipeline, **kwargs) -> Path           (one inference + encode)
  - write_mp4, to_uint8_rgb                              (encoding helpers)

Design:
  - open_mesh() is a context manager: enter once, serve many generate_video()
    calls, exit on shutdown. Fabric + mesh are torn down in its finally.
  - generate_video() is pure-compute: takes a live pipeline, returns a Path.
    No device setup, no teardown. Safe to call many times on one pipeline.
  - Everything is thread-blocking. Callers that need async wrap in
    asyncio.to_thread(...).
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import os
import pathlib
import sys
import time
from fractions import Fraction
from typing import Optional

import numpy as np
import PIL.Image


CONFIGS = {
    "2x2sp0tp1": dict(
        mesh_shape=(2, 2), sp_axis=0, tp_axis=1, num_links=2, dynamic_load=False, is_fsdp=True, fabric="line"
    ),
    "2x4sp0tp1": dict(
        mesh_shape=(2, 4), sp_axis=0, tp_axis=1, num_links=1, dynamic_load=True, is_fsdp=True, fabric="line"
    ),
    "bh_2x4sp1tp0": dict(
        mesh_shape=(2, 4), sp_axis=1, tp_axis=0, num_links=2, dynamic_load=True, is_fsdp=False, fabric="line"
    ),
    "wh_4x8sp1tp0": dict(
        mesh_shape=(4, 8), sp_axis=1, tp_axis=0, num_links=4, dynamic_load=False, is_fsdp=True, fabric="ring"
    ),
    "bh_4x8sp1tp0": dict(
        mesh_shape=(4, 8), sp_axis=1, tp_axis=0, num_links=2, dynamic_load=False, is_fsdp=False, fabric="line"
    ),
}

# (width, height). The HTTP server uses (height, width) in its own map.
RESOLUTIONS = {"480p": (832, 480), "720p": (1280, 720)}

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Inference defaults mirrored from the pytest harness.
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_GUIDANCE_SCALE_2 = 3.5
DEFAULT_SEED = 42
DEFAULT_FPS = 16

# Wan's VAE temporal scale factor. num_frames must satisfy
# (num_frames - 1) % NUM_FRAMES_STEP == 0 (see pipeline_wan.py:835).
# Valid values: 1, 5, 9, 13, ..., 81, 85, 89, ..., 121.
NUM_FRAMES_STEP = 4


def round_up_num_frames(n: int, step: int = NUM_FRAMES_STEP) -> int:
    """Smallest m >= n with (m - 1) % step == 0. Returns n if already valid."""
    if n < 1:
        raise ValueError(f"num_frames must be positive; got {n}")
    if (n - 1) % step == 0:
        return n
    return ((n - 1) // step + 1) * step + 1


@contextlib.contextmanager
def tt_device_lock(path: str, timeout: float):
    """Exclusive flock on the TT device. Mirrors the pytest fixture."""
    lock_dir = os.path.dirname(path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    f = open(path, "a+")
    start = time.monotonic()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                pass
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"TT device lock not acquired within {timeout}s ({path})")
            time.sleep(1.0)
        try:
            f.truncate(0)
            f.write(f"{os.getpid()}\n")
            f.flush()
        except OSError:
            pass
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                pass
        f.close()


@contextlib.contextmanager
def open_mesh(config_name: str, use_lock: bool = True):
    """Set fabric, open parent mesh, carve submesh. Clean up on exit.

    Yields (mesh_device, cfg_dict). cfg_dict is the CONFIGS[config_name]
    entry so callers dont have to re-lookup.

    Designed to be entered once by long-lived processes (server lifespan)
    and also once per CLI invocation. Safe to call pipeline(...) many
    times while inside the with-block.
    """
    import ttnn

    cfg = CONFIGS[config_name]
    req_shape = cfg["mesh_shape"]

    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if cfg["fabric"] == "ring" else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )

    parent = None
    submesh = None
    if use_lock:
        lock_path = os.environ.get("TT_DEVICE_LOCK_PATH", "/tmp/tt_device.lock")
        lock_timeout = float(os.environ.get("TT_DEVICE_LOCK_TIMEOUT", "60"))
        lock_ctx: contextlib.AbstractContextManager = tt_device_lock(lock_path, lock_timeout)
    else:
        lock_ctx = contextlib.nullcontext()

    try:
        with lock_ctx:
            parent = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(req_shape),
                dispatch_core_config=ttnn.DispatchCoreConfig(type=None, axis=None, fabric_tensix_config=None),
            )
            submesh = parent.create_submesh(ttnn.MeshShape(*req_shape))
            yield submesh, cfg
    finally:
        if submesh is not None:
            ttnn.close_mesh_device(submesh)
        if parent is not None:
            ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def create_pipeline(mesh_device, cfg, *, target_height: int, target_width: int, num_frames: int):
    """Build a WanPipelineI2V bound to `mesh_device`.

    Separated from open_mesh() so the server can build the pipeline inside
    the lifespan startup without spreading tt-metal imports across files.
    Call once per mesh_device and reuse for many generate_video() calls.
    """
    import ttnn
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V

    return WanPipelineI2V.create_pipeline(
        mesh_device=mesh_device,
        sp_axis=cfg["sp_axis"],
        tp_axis=cfg["tp_axis"],
        num_links=cfg["num_links"],
        dynamic_load=cfg["dynamic_load"],
        topology=(ttnn.Topology.Ring if cfg["fabric"] == "ring" else ttnn.Topology.Linear),
        is_fsdp=cfg["is_fsdp"],
        target_height=target_height,
        target_width=target_width,
        num_frames=num_frames,
    )


def write_mp4(video_u8: np.ndarray, path: str, fps: int = DEFAULT_FPS) -> None:
    """Encode (T, H, W, 3) uint8 RGB to H.264 MP4, in-process via PyAV.

    Sets vf.pts and stream.time_base explicitly. Without PTS, muxed
    packets share a timestamp and playback stalls on frame 0.
    """
    import av

    video_u8 = np.ascontiguousarray(video_u8)
    _, H, W, _ = video_u8.shape
    H2, W2 = H - (H % 2), W - (W % 2)
    if (H2, W2) != (H, W):
        video_u8 = np.ascontiguousarray(video_u8[:, :H2, :W2, :])

    tb = Fraction(1, fps)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height = W2, H2
        stream.pix_fmt = "yuv420p"
        stream.time_base = tb
        stream.codec_context.time_base = tb
        stream.codec_context.framerate = Fraction(fps, 1)

        for i, frame_rgb in enumerate(video_u8):
            vf = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            vf.pts = i
            vf.time_base = tb
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    scale = 255.0 if float(frames.max()) <= 1.0 else 1.0
    return np.clip(frames * scale, 0, 255).round().astype(np.uint8)


def generate_video(
    pipeline,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    first_image,
    last_image=None,
    image_prompts=None,
    num_frames: int = 81,
    height: int,
    width: int,
    encode_height: Optional[int] = None,
    encode_width: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    guidance: Optional[float] = None,
    guidance_2: Optional[float] = None,
    fps: int = DEFAULT_FPS,
    out_path,
    save_npy: bool = False,
    log=print,
) -> pathlib.Path:
    """Run a single Wan inference on `pipeline` and encode the MP4.

    Either (first_image[, last_image]) or image_prompts may be given:
      - first_image: PIL.Image.Image or path-like
      - last_image:  PIL.Image.Image or path-like or None
      - image_prompts: iterable of (PIL.Image | path-like, frame_pos:int)

    None-valued steps / seed / guidance / guidance_2 / negative_prompt
    fall back to the module defaults. This matches the server's wire
    protocol where optional fields arrive as None.
    """
    import torch
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt

    def _as_pil(x):
        if isinstance(x, PIL.Image.Image):
            return x.convert("RGB")
        return PIL.Image.open(x).convert("RGB")

    if image_prompts is not None:
        prompts_list = [ImagePrompt(image=_as_pil(img), frame_pos=int(fp)) for img, fp in image_prompts]
        if not prompts_list:
            raise ValueError("image_prompts must be non-empty")
    else:
        if first_image is None:
            raise ValueError("first_image is required when image_prompts is not provided")
        prompts_list = [ImagePrompt(image=_as_pil(first_image), frame_pos=0)]
        if last_image is not None:
            prompts_list.append(ImagePrompt(image=_as_pil(last_image), frame_pos=num_frames - 1))

    neg = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
    steps_v = DEFAULT_NUM_INFERENCE_STEPS if steps is None else int(steps)
    seed_v = DEFAULT_SEED if seed is None else int(seed)
    g_v = DEFAULT_GUIDANCE_SCALE if guidance is None else float(guidance)
    g2_v = DEFAULT_GUIDANCE_SCALE_2 if guidance_2 is None else float(guidance_2)

    log(
        f"Running inference: {width}x{height}, {num_frames} frames, "
        f"{steps_v} steps, seed={seed_v}, g={g_v}/{g2_v}, "
        f"conditioning_frames={[p.frame_pos for p in prompts_list]}",
        file=sys.stderr,
    )

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            image_prompt=prompts_list,
            negative_prompt=neg,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps_v,
            seed=seed_v,
            guidance_scale=g_v,
            guidance_scale_2=g2_v,
        )

    frames = result.frames[0]  # (T, H, W, 3)
    log(
        f"Inference done. frames.shape={frames.shape}, dtype={frames.dtype}, "
        f"range=[{float(frames.min()):.3f}, {float(frames.max()):.3f}]",
        file=sys.stderr,
    )
    del result
    gc.collect()

    out_path = pathlib.Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_u8 = to_uint8_rgb(frames)

    # Optional in-process resize before mp4 encoding (avoids spawning ffmpeg
    # from this huge parent process; fork() would ENOMEM like earlier).
    enc_h = height if encode_height is None else int(encode_height)
    enc_w = width if encode_width is None else int(encode_width)
    if (enc_h, enc_w) != (int(height), int(width)):
        log(f"Resizing frames {width}x{height} -> {enc_w}x{enc_h} via PIL", file=sys.stderr)
        resized = np.empty((video_u8.shape[0], enc_h, enc_w, 3), dtype=np.uint8)
        for i, frame in enumerate(video_u8):
            resized[i] = np.asarray(PIL.Image.fromarray(frame).resize((enc_w, enc_h), PIL.Image.LANCZOS))
        video_u8 = resized

    if save_npy:
        npy_path = str(out_path) + ".npy"
        np.save(npy_path, video_u8)
        log(f"Saved raw frames to: {npy_path}", file=sys.stderr)

    write_mp4(video_u8, str(out_path), fps=fps)
    return out_path
