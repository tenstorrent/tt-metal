#!/usr/bin/env python3
"""How does CPU speech-tokenizer decode scale with torch threads?

The benchmark I ran measured TTFT on a 64-core host where the single server
instance had torch's default 32 threads to itself. Production runs one media
server per chip -- 32 instances on this box -- so each instance realistically
gets 1-2 cores, not 32. This measures the same decode at 1/2/4/8/16/32 threads so
the production TTFT can be estimated instead of guessed.

CPU only: no device, no ttnn ops, safe to run any time.

Usage:
    python measure_cpu_decode_threads.py
"""

import os
import time

# Thread limits must be set before torch initialises its pools.
os.environ.setdefault("OMP_NUM_THREADS", "32")

import torch

from models.demos.qwen3_tts.reference.functional import (
    SpeechTokenizerDecoderConfig,
    speech_tokenizer_decoder_forward,
)

REF_FRAMES = 51  # the built-in 'jim' prompt (4.01s @ 12.5fps)
# ~228 generated frames is what the benchmark implies (RTR 0.290 at TTFT 5.285s
# => ~18s of audio), plus a short case for contrast.
GEN_FRAME_CASES = [21, 228]
THREAD_COUNTS = [1, 2, 4, 8, 16, 32]


def load_decoder_weights():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "speech_tokenizer/model.safetensors")
    sd = load_file(path)
    return {k[len("decoder.") :]: v.float() for k, v in sd.items() if k.startswith("decoder.")}


def main():
    print(f"host cores: {os.cpu_count()}", flush=True)
    weights = load_decoder_weights()
    cfg = SpeechTokenizerDecoderConfig()

    for gen in GEN_FRAME_CASES:
        total = REF_FRAMES + gen
        # decode_icl_audio decodes cat([ref, generated]) then cuts the reference.
        codes = torch.zeros(total, 16, dtype=torch.long)
        token_ids = codes.T.unsqueeze(0)
        audio_s = gen / 12.5
        print(f"\n=== ref{REF_FRAMES} + gen{gen} = {total} frames "
              f"({audio_s:.1f}s of output audio) ===", flush=True)
        print(f"{'threads':>8}  {'decode_s':>9}  {'vs 32thr':>9}", flush=True)

        baseline = None
        for n in THREAD_COUNTS:
            torch.set_num_threads(n)
            with torch.no_grad():  # warm once so we time steady state
                speech_tokenizer_decoder_forward(token_ids.clone(), weights, cfg)
                t = time.perf_counter()
                speech_tokenizer_decoder_forward(token_ids.clone(), weights, cfg)
                dt = time.perf_counter() - t
            if n == 32:
                baseline = dt
            print(f"{n:>8}  {dt:>9.2f}  {'' if baseline is None else f'{dt/baseline:>8.2f}x'}",
                  flush=True)

        # Re-report relative to the 32-thread figure now that we have it.
        if baseline:
            print(f"  (32-thread decode = {baseline:.2f}s; "
                  f"1-thread is the realistic per-instance budget at 32 instances)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
