#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo with Metal Trace — traces the DIT denoising loop for fast inference.

The DIT runs 9 steps per prompt and dominates end-to-end latency.  Metal Trace
eliminates host dispatch overhead for every DIT call by recording the TTNN op
graph once and replaying it from device-side command buffers.

Text encoder and VAE decoder each run once per prompt and are NOT traced.

Usage:
    python zit.py
    python zit.py --steps 9 --seed 42
"""

import argparse
import copy
import os
import sys
import time

# Make imports work regardless of cwd (same pattern as server.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

from text_encoder.model_ttnn import TextEncoderTTNN
from dit.model_ttnn import ZImageTransformerTTNN
from vae.model_ttnn import VaeDecoderTTNN

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


# ── Tensor helpers ────────────────────────────────────────────────────────────


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


def _copy_to_persistent(host_pt, persistent_tt, dtype=ttnn.DataType.BFLOAT16):
    """Write a CPU tensor into every per-device shard of a persistent mesh tensor.

    Uses per-device iteration so that all 4 BH devices get the new data
    (plain copy_host_to_device_tensor only reaches device 0).
    """
    host_tt = ttnn.from_torch(
        host_pt.bfloat16() if dtype == ttnn.DataType.BFLOAT16 else host_pt,
        dtype=dtype,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    for shard in ttnn.get_device_tensors(persistent_tt):
        ttnn.copy_host_to_device_tensor(host_tt, shard, cq_id=0)


# ── Scheduler helper ─────────────────────────────────────────────────────────


def _compute_mu(
    h=IMG_LATENT_H,
    w=IMG_LATENT_W,
    base_seq=256,
    max_seq=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    seq = (h // 2) * (w // 2)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq * m + (base_shift - m * base_seq)


def _make_scheduler(template, steps):
    scheduler = copy.deepcopy(template)
    mu = _compute_mu()
    try:
        scheduler.set_timesteps(steps, mu=mu)
    except TypeError:
        scheduler.set_timesteps(steps)
    return scheduler


# ── ZImageTurbo class ─────────────────────────────────────────────────────────


class ZImageTurbo:
    """Z-Image-Turbo pipeline with Metal Trace on the DIT denoising loop."""

    def __init__(self):
        t0 = time.time()

        print("[1/5] Loading CPU components ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        self.vae_processor = VaeImageProcessor(vae_scale_factor=16)
        self.scheduler_template = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.scheduler_template.sigma_min = 0.0

        print("[2/5] Opening TTNN (1,4) mesh device ...")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        self.mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape((1, 4)),
            l1_small_size=1 << 15,
            trace_region_size=60_000_000,
        )
        self.mesh_device.enable_program_cache()

        print("[3/5] Building Text Encoder ...")
        self.te = TextEncoderTTNN(self.mesh_device)

        print("[4/5] Building Transformer (DIT) ...")
        self.tr = ZImageTransformerTTNN(self.mesh_device)

        print("[5/5] Loading TTNN VAE decoder ...")
        self.vae = VaeDecoderTTNN(self.mesh_device)

        self._trace_id = None
        self._lat_buf = None
        self._ts_buf = None
        self._output_ref = None

        print(f"Model loading: {(time.time() - t0) * 1000:.0f} ms\n")

    # ── Text encoding (runs without trace) ────────────────────────────────────

    def _encode_prompt(self, prompt):
        """Tokenize + encode → [1, CAP_TOKENS, 2560] BF16 CPU tensor."""
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
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

        tt_input_ids = _to_device_int32(input_ids, self.mesh_device)
        tt_cap_out = self.te(tt_input_ids)
        cap_cpu = _tt_to_torch(tt_cap_out, self.mesh_device)[:CAP_TOKENS].bfloat16()
        # Free device tensors so they don't occupy addresses the DIT trace needs.
        ttnn.deallocate(tt_input_ids, force=True)
        ttnn.deallocate(tt_cap_out, force=True)
        return cap_cpu.unsqueeze(0)  # [1, 32, 2560]

    # ── VAE decode (runs without trace) ───────────────────────────────────────

    def _decode_latents(self, latents):
        """VAE decode on TTNN -> PIL image."""
        image_tensor = self.vae(latents)
        # Force GC to free VAE intermediate device tensors before next DIT trace.
        # The trace bakes in memory addresses; any stray device allocation at those
        # addresses would be overwritten during trace replay, corrupting VAE state.
        import gc

        gc.collect()
        return self.vae_processor.postprocess(image_tensor, output_type="pil")[0]

    # ── Warmup: compile + capture DIT trace ───────────────────────────────────

    def warmup(self, prompt="a cat sitting on a mat", steps=9, seed=42):
        """Compile the DIT, capture Metal Trace, and run one full generation.

        After this call, generate() uses the cached trace for all DIT calls.
        """
        print("=" * 72)
        print("WARMUP: compile all programs → capture DIT trace → generate")
        print("=" * 72)
        t_total = time.time()

        # ── Phase 1: Compile ALL programs (TE, DIT, VAE) BEFORE trace capture.
        # The trace bakes in device memory addresses.  Any program compiled AFTER
        # trace capture may place its binary at an address the trace uses for
        # intermediates; the next trace replay then overwrites the binary and the
        # affected model hangs.

        # 1a) Compile TE programs.
        t0 = time.time()
        cap_cpu = self._encode_prompt(prompt)
        print(f"  TE compile: {(time.time() - t0) * 1000:.0f} ms")

        # 1b) Compile DIT programs.
        self.tr.set_cap_feats(cap_cpu)
        torch.manual_seed(seed)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        t0_step = scheduler.timesteps[0]
        t_norm_0 = max((1000.0 - float(t0_step)) / 1000.0, 1e-3)
        lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
        ts_pt = torch.tensor([t_norm_0], dtype=torch.bfloat16)

        self._lat_buf = _to_device_bf16(lat_pt, self.mesh_device)
        self._ts_buf = _to_device_bf16(ts_pt, self.mesh_device)

        t0 = time.time()
        compile_out = self.tr._forward_impl([self._lat_buf], self._ts_buf)
        print(f"  DIT compile: {(time.time() - t0) * 1000:.0f} ms")

        # Read DIT output so we can feed it to VAE for compilation.
        dit_out_cpu = _tt_to_torch(compile_out[0], self.mesh_device)
        dit_out_cpu = dit_out_cpu.squeeze(1).unsqueeze(0).bfloat16()
        compile_latents = scheduler.step(-dit_out_cpu.float(), scheduler.timesteps[0], latents, return_dict=False)[0]

        for t in compile_out:
            ttnn.deallocate(t, force=True)

        # 1c) Compile VAE programs (first decode, pays consteval cost).
        t0 = time.time()
        _warmup_image = self._decode_latents(compile_latents)
        print(f"  VAE compile: {(time.time() - t0) * 1000:.0f} ms")

        # ── Phase 2: Capture DIT trace (all programs already compiled).
        t0 = time.time()
        self._trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        trace_out = self.tr._forward_impl([self._lat_buf], self._ts_buf)
        ttnn.end_trace_capture(self.mesh_device, self._trace_id, cq_id=0)
        self._output_ref = trace_out[0]
        print(f"  DIT trace capture: {(time.time() - t0) * 1000:.0f} ms")

        # ── Phase 3: Full warmup generation using the trace.
        torch.manual_seed(seed)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        # The scheduler produces N timesteps but the last one (t=0) is a no-op
        # (dt=0, so latents += 0). Skip it to save one full DIT execution.
        active_timesteps = scheduler.timesteps[:-1]
        t0 = time.time()
        for i, t in enumerate(active_timesteps):
            t_norm = max((1000.0 - float(t)) / 1000.0, 1e-3)
            lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
            ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)

            _copy_to_persistent(lat_pt, self._lat_buf)
            _copy_to_persistent(ts_pt, self._ts_buf)

            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)

            out = _tt_to_torch(self._output_ref, self.mesh_device)
            out = out.squeeze(1).unsqueeze(0).bfloat16()
            latents = scheduler.step(-out.float(), t, latents, return_dict=False)[0]
        print(f"  warmup denoising ({len(active_timesteps)} steps): {(time.time() - t0) * 1000:.0f} ms")

        t0 = time.time()
        image = self._decode_latents(latents)
        print(f"  VAE decode: {(time.time() - t0) * 1000:.0f} ms")

        print(f"  WARMUP TOTAL: {(time.time() - t_total) * 1000:.0f} ms")
        print("=" * 72)
        print("Trace captured. Ready for fast generation.\n")
        return image

    # ── Fast generation using cached trace ────────────────────────────────────

    def generate(self, prompt, steps=9, seed=42):
        """Generate an image using the cached DIT trace.

        Must call warmup() first. Returns a PIL Image.
        """
        if self._trace_id is None:
            raise RuntimeError("Call warmup() before generate().")

        t_total = time.time()

        # 1) Text encoding (TE runs normally).
        t0 = time.time()
        cap_cpu = self._encode_prompt(prompt)
        te_ms = (time.time() - t0) * 1000
        print(f"  text encoding: {te_ms:.0f} ms")

        # 2) Copy new caption features to the DIT's persistent buffer.
        #    MUST NOT call set_cap_feats() — that allocates a new tensor and
        #    breaks the trace which baked in the old address.
        t0 = time.time()
        _copy_to_persistent(cap_cpu, self.tr._cap_feats_buf)
        print(f"  cap_feats copy: {(time.time() - t0) * 1000:.0f} ms")

        # 3) Init noise + scheduler.
        torch.manual_seed(seed)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        # 4) Denoising loop — DIT via trace replay.
        # Skip last timestep (t=0, dt=0 → no-op). 8 DIT forwards with 9-step schedule.
        active_timesteps = scheduler.timesteps[:-1]
        num_steps = len(active_timesteps)
        step_times = []
        for i, t in enumerate(active_timesteps):
            t0 = time.time()

            t_norm = max((1000.0 - float(t)) / 1000.0, 1e-3)
            lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()  # [16,1,64,64]
            ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)  # [1]

            _copy_to_persistent(lat_pt, self._lat_buf)
            _copy_to_persistent(ts_pt, self._ts_buf)

            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)

            out = _tt_to_torch(self._output_ref, self.mesh_device)
            out = out.squeeze(1).unsqueeze(0).bfloat16()
            latents = scheduler.step(-out.float(), t, latents, return_dict=False)[0]

            elapsed = (time.time() - t0) * 1000
            step_times.append(elapsed)
            print(f"  step {i + 1}/{num_steps}: {elapsed:.0f} ms")

        # 5) VAE decode (runs normally).
        t0 = time.time()
        image = self._decode_latents(latents)
        vae_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - t_total) * 1000
        steady = step_times[1:] if len(step_times) > 1 else step_times
        avg_steady = sum(steady) / len(steady) if steady else 0
        print(f"  VAE: {vae_ms:.0f} ms  |  total: {total_ms:.0f} ms  " f"|  steady-state step: {avg_steady:.0f} ms avg")
        return image


# ── Interactive server ────────────────────────────────────────────────────────

DEFAULT_STEPS = 9
DEFAULT_SEED = 42
OUTPUT_DIR = "outputs"
WARMUP_PROMPT = "a cat sitting on a mat"


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo interactive server with Metal Trace",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Denoising steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed (default: {DEFAULT_SEED}); " "each prompt uses seed + idx so repeats yield new images",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        import readline  # noqa: F401  -- enables history/editing inside input()
    except ImportError:
        pass

    # 1) Load models + warmup (compile all programs + capture DIT trace).
    pipeline = ZImageTurbo()

    bar = "=" * 72
    print(bar)
    print(f"Warming up with dummy prompt {WARMUP_PROMPT!r}")
    print("(first run is slow: compile all programs + Metal Trace capture)")
    print(bar)

    t0 = time.time()
    warmup_image = pipeline.warmup(steps=args.steps, seed=args.seed)

    path = os.path.join(OUTPUT_DIR, "out_0.png")
    warmup_image.save(path)
    print(f"  END-TO-END: {time.time() - t0:.2f} s  |  {os.path.abspath(path)}\n")

    # 2) REPL loop.
    print(bar)
    print("Ready. Type a prompt and press ENTER to generate.")
    print("Ctrl-D or Ctrl-C to exit.")
    print(bar)

    idx = 1
    while True:
        try:
            prompt = input(f"\nprompt [{idx}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not prompt:
            continue

        try:
            path = os.path.join(OUTPUT_DIR, f"out_{idx}.png")
            t_start = time.time()
            image = pipeline.generate(
                prompt,
                steps=args.steps,
                seed=args.seed + idx,
            )
            image.save(path)
            elapsed = time.time() - t_start
            print(f"  END-TO-END: {elapsed:.2f} s  |  {os.path.abspath(path)}\n")
            idx += 1
        except KeyboardInterrupt:
            print("\n  Interrupted during generation; shutting down.")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}\n")


if __name__ == "__main__":
    main()
