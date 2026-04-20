#!/usr/bin/env python3
"""Standalone Wan2.2 I2V generator for Tenstorrent hardware (CLI).

Thin wrapper around wan_i2v_core.{open_mesh, create_pipeline, generate_video}.
Run from the tt-metal repo root with the python_env activated:

    source python_env/bin/activate
    python wan_i2v_generate.py \
        --first-image start.png \
        --prompt "the camera dollies into a sunlit garden" \
        --output garden.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import PIL.Image

from wan_i2v_core import (
    CONFIGS,
    DEFAULT_NEGATIVE_PROMPT,
    RESOLUTIONS,
    create_pipeline,
    generate_video,
    open_mesh,
    round_up_num_frames,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--first-image", required=True, help="Path to the first-frame image.")
    p.add_argument(
        "--last-image",
        default=None,
        help="Path to the last-frame image. For FLF2V-style conditioning, pass a DIFFERENT "
        "image from --first-image. Conditioning on the same image tends to produce static output.",
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

    if args.num_frames < 1:
        sys.exit("--num-frames must be positive")
    rounded = round_up_num_frames(args.num_frames)
    if rounded != args.num_frames:
        print(
            f"[num_frames] rounded {args.num_frames} -> {rounded} "
            "to satisfy (n - 1) %% 4 == 0 (Wan temporal latent alignment)",
            file=sys.stderr,
        )
        args.num_frames = rounded

    output = pathlib.Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = RESOLUTIONS[args.resolution]

    first_img = PIL.Image.open(first).convert("RGB")
    last_img = PIL.Image.open(last_path).convert("RGB") if last_path else None

    with open_mesh(args.config, use_lock=not args.no_lock) as (mesh_device, cfg):
        pipeline = create_pipeline(
            mesh_device,
            cfg,
            target_height=height,
            target_width=width,
            num_frames=args.num_frames,
        )

        generate_video(
            pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            first_image=first_img,
            last_image=last_img,
            num_frames=args.num_frames,
            height=height,
            width=width,
            steps=args.steps,
            seed=args.seed,
            guidance=args.guidance,
            guidance_2=args.guidance_2,
            fps=args.fps,
            out_path=output,
            save_npy=args.save_npy,
        )

        print(f"Saved video to: {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
