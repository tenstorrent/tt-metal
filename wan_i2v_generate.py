#!/usr/bin/env python3
"""Standalone Wan2.2 I2V generator for Tenstorrent hardware.

Runs end-to-end without pytest:
  - Acquires the TT device lock (optional, opt-out with --no-lock).
  - Configures 1D / 1D-ring fabric.
  - Opens the parent mesh, carves out the requested submesh.
  - Builds WanPipelineI2V, runs inference, tears everything down.
  - Encodes output with PyAV (libav linked in-process, no fork).

Invoke from the tt-metal repo root with the python_env activated:

    source python_env/bin/activate
    python wan_i2v_generate.py \
        --first-image start.png \
        --prompt "the camera dollies into a sunlit garden" \
        --output garden.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import os
import pathlib
import sys
import time
from fractions import Fraction

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

RESOLUTIONS = {"480p": (832, 480), "720p": (1280, 720)}

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


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
def open_mesh(config_name: str, use_lock: bool):
    """Open parent mesh, carve submesh, ensure clean teardown."""
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


def write_mp4(video_u8: np.ndarray, path: str, fps: int = 16) -> None:
    """Encode (T, H, W, 3) uint8 RGB to H.264 MP4, in-process via PyAV.

    Critical: set vf.pts and stream.time_base. Without PTS, muxed packets
    share a timestamp and playback stalls on frame 0 for the whole clip.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--first-image", required=True, help="Path to the first-frame image.")
    p.add_argument(
        "--last-image",
        default=None,
        help=(
            "Path to the last-frame image (optional). NOTE: Wan2.2-I2V-A14B has "
            "image_dim=null and is not a FLF2V checkpoint; results with --last-image "
            "on this model will likely be degenerate."
        ),
    )
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt. Defaults to the Wan2.2 recommended Chinese negatives.",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Must satisfy (n - 1) %% 4 == 0. The pipeline warmup hardcodes 81.",
    )
    p.add_argument("--resolution", choices=RESOLUTIONS, default="480p")
    p.add_argument("--config", choices=CONFIGS, default="bh_4x8sp1tp0")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--guidance-2", type=float, default=3.5)
    p.add_argument("--fps", type=int, default=16, help="Output video fps. Wan is trained at 16.")
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the raw (T, H, W, 3) uint8 numpy array next to the .mp4.",
    )
    p.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip the TT device lock. Use when no other TT job is running.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    first = pathlib.Path(args.first_image).expanduser().resolve()
    if not first.is_file():
        sys.exit(f"First image not found: {first}")

    last_path: pathlib.Path | None = None
    if args.last_image:
        last_path = pathlib.Path(args.last_image).expanduser().resolve()
        if not last_path.is_file():
            sys.exit(f"Last image not found: {last_path}")

    if (args.num_frames - 1) % 4 != 0:
        sys.exit(f"--num-frames must satisfy (n - 1) % 4 == 0; got {args.num_frames}")
    if args.num_frames < 1:
        sys.exit("--num-frames must be positive")

    output = pathlib.Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = RESOLUTIONS[args.resolution]

    first_img = PIL.Image.open(first).convert("RGB")
    last_img = PIL.Image.open(last_path).convert("RGB") if last_path else None

    import torch
    import ttnn
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V

    with open_mesh(args.config, use_lock=not args.no_lock) as (mesh_device, cfg):
        pipeline = WanPipelineI2V.create_pipeline(
            mesh_device=mesh_device,
            sp_axis=cfg["sp_axis"],
            tp_axis=cfg["tp_axis"],
            num_links=cfg["num_links"],
            dynamic_load=cfg["dynamic_load"],
            topology=(ttnn.Topology.Ring if cfg["fabric"] == "ring" else ttnn.Topology.Linear),
            is_fsdp=cfg["is_fsdp"],
            target_height=height,
            target_width=width,
            num_frames=args.num_frames,
        )

        image_prompt = [ImagePrompt(image=first_img, frame_pos=0)]
        if last_img is not None:
            image_prompt.append(ImagePrompt(image=last_img, frame_pos=args.num_frames - 1))

        print(
            f"Running inference: {width}x{height}, {args.num_frames} frames, "
            f"{args.steps} steps, seed={args.seed}, g={args.guidance}/{args.guidance_2}",
            file=sys.stderr,
        )

        with torch.no_grad():
            result = pipeline(
                prompt=args.prompt,
                image_prompt=image_prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                seed=args.seed,
                guidance_scale=args.guidance,
                guidance_scale_2=args.guidance_2,
            )

        frames = result.frames[0]
        print(
            f"Inference done. frames.shape={frames.shape}, dtype={frames.dtype}, "
            f"range=[{float(frames.min()):.3f}, {float(frames.max()):.3f}]",
            file=sys.stderr,
        )
        del result
        gc.collect()

        video_u8 = to_uint8_rgb(frames)

        if args.save_npy:
            npy_path = str(output) + ".npy"
            np.save(npy_path, video_u8)
            print(f"Saved raw frames to: {npy_path}", file=sys.stderr)

        write_mp4(video_u8, str(output), fps=args.fps)
        print(f"Saved video to: {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
