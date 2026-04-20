#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo demo — generate 512×512 images from text prompts.

Hardware: 4× Blackhole P150, tensor-parallel across (1,4) mesh.
Models:   text encoder + transformer + VAE decoder all on TTNN.
Speed:    VAE consteval runs on first decode (~330 s); subsequent decodes ~1.4 s.

Single prompt:
    python generate.py "a misty mountain lake at dawn"
    python generate.py "a robot in an art studio" --output robot.png --seed 0

Multiple prompts (loaded once, run back to back):
    python generate.py "a cat" "a dog" "a fox"
    python generate.py "a cat" "a dog" --output-dir results/ --seed 42 --steps 9
    python generate.py --prompts-file prompts.txt --output-dir results/ --seed 42
    python generate.py --prompts-file prompts.txt --output-dir results/

prompts.txt format: one prompt per line, blank lines and # comments ignored.
"""

import argparse
import os
import re
import sys
import time

# Make imports work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

from text_encoder.model_ttnn import TextEncoderTTNN
from dit.model_ttnn import ZImageTransformerTTNN
from vae.model_ttnn import VaeDecoderTTNN

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128  # caption tokens (baked into compiled model)
IMG_LATENT_H = 64  # 512 px / 8 (VAE scale)
IMG_LATENT_W = 64
LATENT_CHANNELS = 16
DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


# ── Tensor helpers ─────────────────────────────────────────────────────────────


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


# ── Scheduler ─────────────────────────────────────────────────────────────────


def _compute_mu(h=IMG_LATENT_H, w=IMG_LATENT_W, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
    seq = (h // 2) * (w // 2)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq * m + (base_shift - m * base_seq)


# ── Output filename ────────────────────────────────────────────────────────────


def _output_path(prompt, index, output_dir):
    """Build an output filename from the prompt and index."""
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")[:40]
    return os.path.join(output_dir, f"{index:02d}_{slug}.png")


# ── Model loading (once, shared across all prompts) ────────────────────────────


def open_mesh_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 4)),
        l1_small_size=1 << 15,
        trace_region_size=50_000_000,
    )
    device.enable_program_cache()
    return device


class Models:
    """Loaded models, shared across all prompts in a run."""

    def __init__(self):
        print("[1/5] Loading CPU components ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        self.vae_processor = VaeImageProcessor(vae_scale_factor=16)
        self.scheduler_template = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.scheduler_template.sigma_min = 0.0

        print("[2/5] Opening TTNN (1,4) mesh device ...")
        self.mesh_device = open_mesh_device()

        print("[3/5] Building Text Encoder ...")
        self.te = TextEncoderTTNN(self.mesh_device)

        print("[4/5] Building Transformer ...")
        self.tr = ZImageTransformerTTNN(self.mesh_device)

        print("[5/5] Loading TTNN VAE decoder (weights from HuggingFace) ...")
        self.vae_tt = VaeDecoderTTNN(self.mesh_device)
        print()

    def encode_prompt(self, prompt):
        """Tokenize and encode a prompt → [1, CAP_TOKENS, 2560] BF16 CPU tensor."""
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=CAP_TOKENS,
            return_tensors="pt",
        )["input_ids"]
        tt_cap_out = self.te(_to_device_int32(input_ids, self.mesh_device))
        cap_cpu = _tt_to_torch(tt_cap_out, self.mesh_device)[:CAP_TOKENS].bfloat16()
        return cap_cpu.unsqueeze(0)  # [1, 32, 2560] on CPU

    def decode_latents(self, latents):
        """VAE decode on TTNN → PIL image."""
        image_tensor = self.vae_tt(latents)
        return self.vae_processor.postprocess(image_tensor, output_type="pil")[0]


# ── Per-prompt generation ──────────────────────────────────────────────────────


def _run_one(models, prompt, steps, seed, output_path, prompt_index, total_prompts):
    """Generate one image. Returns wall-clock ms."""
    import copy

    torch.manual_seed(seed)

    header = f"[{prompt_index}/{total_prompts}]" if total_prompts > 1 else ""
    print(f"{header} Prompt: {prompt!r}  (seed={seed}, steps={steps})")

    t_start = time.time()

    # Text encoding
    t0 = time.time()
    cap_cpu = models.encode_prompt(prompt)
    models.tr.set_cap_feats(cap_cpu)
    print(f"  text encoding: {(time.time()-t0)*1000:.0f} ms")

    # Fresh noise + scheduler (independent per prompt)
    latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
    scheduler = copy.deepcopy(models.scheduler_template)
    mu = _compute_mu()
    try:
        scheduler.set_timesteps(steps, mu=mu)
    except TypeError:
        scheduler.set_timesteps(steps)

    # Denoising — skip last timestep (t=0, dt=0 → no-op). 8 DIT forwards with 9-step schedule.
    active_timesteps = scheduler.timesteps[:-1]
    num_steps = len(active_timesteps)
    step_times = []
    for i, t in enumerate(active_timesteps):
        t0 = time.time()
        t_norm = max((1000.0 - float(t)) / 1000.0, 1e-3)
        tt_t = _to_device_bf16(torch.tensor([t_norm], dtype=torch.bfloat16), models.mesh_device)
        tt_lat = _to_device_bf16(latents.squeeze(0).unsqueeze(1).bfloat16(), models.mesh_device)

        tt_out = models.tr([tt_lat], tt_t)[0]
        out = _tt_to_torch(tt_out, models.mesh_device).squeeze(1).unsqueeze(0).bfloat16()
        latents = scheduler.step(-out.float(), t, latents, return_dict=False)[0]

        elapsed = (time.time() - t0) * 1000
        step_times.append(elapsed)
        print(f"  step {i+1}/{num_steps}: {elapsed:.0f} ms")

    # VAE decode
    t0 = time.time()
    image = models.decode_latents(latents)
    vae_ms = (time.time() - t0) * 1000

    total_ms = (time.time() - t_start) * 1000
    steady = step_times[1:] if len(step_times) > 1 else step_times
    print(
        f"  VAE: {vae_ms:.0f} ms  |  total: {total_ms:.0f} ms  "
        f"|  steady-state step: {sum(steady)/len(steady):.0f} ms avg"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    print(f"  → {output_path}\n")
    return total_ms


# ── Public API ─────────────────────────────────────────────────────────────────


def run(prompts, steps=9, seed=42, output_dir=".", output=None):
    """Generate images for one or more prompts, loading models once.

    Args:
        prompts:    list of prompt strings.
        steps:      denoising steps (same for all prompts).
        seed:       random seed used for all prompts.
        output_dir: directory for output files when len(prompts) > 1.
        output:     explicit output path; only used when len(prompts) == 1.

    Returns:
        List of output file paths.
    """
    if not prompts:
        raise ValueError("No prompts provided.")

    models = Models()
    t_wall = time.time()
    outputs = []

    for i, prompt in enumerate(prompts, start=1):
        if len(prompts) == 1 and output:
            path = output
        else:
            path = _output_path(prompt, i, output_dir)
        _run_one(models, prompt, steps, seed, path, i, len(prompts))
        outputs.append(path)

    if len(prompts) > 1:
        print(f"{'─'*60}")
        print(f"All {len(prompts)} images done in {(time.time()-t_wall):.1f} s")
        print(f"Outputs: {output_dir}/")

    return outputs


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo — text-to-image on 4× Blackhole P150",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python generate.py "a cat"
  python generate.py "a cat" "a dog" "a fox" --output-dir results/
  python generate.py --prompts-file prompts.txt --output-dir results/ --steps 9
""",
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="One or more text prompts (positional)",
    )
    parser.add_argument(
        "--prompts-file",
        metavar="FILE",
        help="Text file with one prompt per line (# comments and blank lines ignored)",
    )
    parser.add_argument(
        "--output",
        default="output.png",
        help="Output file path for a single prompt (default: output.png)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for multiple prompts (default: current dir)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Denoising steps (default: 9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42); same seed used for all prompts",
    )
    args = parser.parse_args()

    prompts = list(args.prompts)

    if args.prompts_file:
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(line)

    if not prompts:
        parser.error("Provide at least one prompt (positional) or --prompts-file.")

    run(
        prompts=prompts,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        output=args.output if len(prompts) == 1 else None,
    )


if __name__ == "__main__":
    main()
