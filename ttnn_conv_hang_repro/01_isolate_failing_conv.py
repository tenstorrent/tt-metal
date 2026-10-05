#!/usr/bin/env python3
"""Case A: find the exact ttnn.conv1d call that deadlocks weight-prepare.

Symptom: warming the qwen3-tts speech decoder at 64 and 128 frames succeeds, but
192 frames deadlocks the device during conv weight-prepare. No talker, no trace,
no 2 command queues involved -- a single process doing nothing but conv1d.

This wraps ttnn.conv1d to print every call's parameters BEFORE invoking it, then
warms 64, 128, 192 in order. 64 and 128 complete; when 192 hangs, the LAST LINE
PRINTED is the exact conv that deadlocked. That line is the bug report.

The hang signature is ~106% CPU on one thread with no progress, uninterruptible
by SIGINT. Killing the process leaves stale ethernet dispatch kernels, so the
board needs `tt-smi -glx_reset_auto` afterwards.

Usage:
    TT_REPRO_DEVICE_ID=0 python 01_isolate_failing_conv.py
"""

import os
import sys
import time

import torch

import ttnn
from models.demos.qwen3_tts.tt import server as api

DEVICE_ID = int(os.environ.get("TT_REPRO_DEVICE_ID", "0"))
# Production serving config from Qwen3TTSConstants. Note the hang reproduces with
# num_command_queues=1 too, so 2CQ is not required to trigger it.
L1_SMALL_SIZE = 32768
TRACE_REGION_SIZE = 512_000_000
NUM_COMMAND_QUEUES = 2

BUCKETS = [int(x) for x in os.environ.get("TT_BUCKETS", "64,128,192").split(",")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_decoder_weights():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "speech_tokenizer/model.safetensors")
    sd = load_file(path)
    return {k[len("decoder.") :]: v.float() for k, v in sd.items() if k.startswith("decoder.")}


def instrument_conv1d():
    """Print every conv1d's parameters before the call, so a hang names itself."""
    original = ttnn.conv1d
    counter = {"n": 0}

    def traced(*args, **kwargs):
        counter["n"] += 1
        desc = (
            f"conv1d #{counter['n']:4d}  "
            f"in_c={kwargs.get('in_channels')} out_c={kwargs.get('out_channels')} "
            f"kernel={kwargs.get('kernel_size')} stride={kwargs.get('stride')} "
            f"pad={kwargs.get('padding')} dilation={kwargs.get('dilation')} "
            f"groups={kwargs.get('groups')} batch={kwargs.get('batch_size')} "
            f"input_length={kwargs.get('input_length')}"
        )
        print(f"  CALLING  {desc}", flush=True)
        out = original(*args, **kwargs)
        print(f"  returned {desc}", flush=True)
        return out

    ttnn.conv1d = traced


def main():
    log(f"device={DEVICE_ID} l1_small={L1_SMALL_SIZE} trace={TRACE_REGION_SIZE} cqs={NUM_COMMAND_QUEUES}")
    log(f"buckets to warm (in order): {BUCKETS}")
    instrument_conv1d()

    device = ttnn.open_device(
        device_id=DEVICE_ID,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=NUM_COMMAND_QUEUES,
    )
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()

    try:
        weights = load_decoder_weights()
        decoder = api.build_device_decoder(device, weights)

        for n in BUCKETS:
            log(f"=== warming bucket {n} ===")
            t = time.perf_counter()
            codes = torch.zeros(n, 16, dtype=torch.long)
            decoder.forward(codes.T.unsqueeze(0))
            log(f"=== bucket {n} OK in {time.perf_counter() - t:.1f}s ===")

        log("ALL BUCKETS WARMED -- bug did not reproduce")
        return 0
    finally:
        log("closing device")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
