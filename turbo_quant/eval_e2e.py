#!/usr/bin/env python3
"""End-to-end Llama-3.1-8B-Instruct inference using TurboQuant KV cache on Wormhole.

NOTE: This script uses TEACHER-FORCED decode throughout (no real prefill).
Each prompt token is fed one at a time with the ground-truth next token forced
as input, so the model never compounds prediction errors during the prompt.
This is useful for KV cache correctness testing, but does not reflect real
deployment latency or generation quality.

For a realistic demo with a single prefill pass followed by free decode, use
eval_e2e_prefill.py instead.

Usage:
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    HF_HOME=/localdev/mtairum/hf \
    TT_CACHE_PATH=/localdev/mtairum/hf/ttnn_cache \
    PYTHONPATH=/localdev/mtairum/tt-metal \
    python turbo_quant/eval_e2e.py [--prompt "..."] [--bits 3] [--max-new-tokens 30]
"""

import argparse
import time

import torch
import ttnn

from models.tt_transformers.tt.common import Mode, copy_host_to_device, sample_host
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from turbo_quant.ttnn_integration import TTNNTurboQuantCache


def build_parser():
    p = argparse.ArgumentParser(description="TurboQuant end-to-end inference")
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--instruct", action="store_true", default=True, help="Use instruct prompt formatting")
    p.add_argument("--num-layers", type=int, default=None, help="Limit to N layers (default: all 32)")
    p.add_argument("--no-turbo-quant", action="store_true", help="Baseline: skip TurboQuant, use standard BFP8 cache")
    p.add_argument("--no-trace", action="store_true", help="Disable TTNN trace (slower, useful for debugging)")
    return p


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ #
    # Device + model setup                                                 #
    # ------------------------------------------------------------------ #
    print("Opening mesh device (1x1)...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    print("Creating ModelArgs...")
    model_args = ModelArgs(
        mesh_device,
        instruct=args.instruct,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        # accuracy mode keeps weights in BF16 where possible
        optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
        cache_hf=True,
    )
    if args.num_layers is not None:
        model_args.n_layers = args.num_layers

    tokenizer = model_args.tokenizer

    # Tokenize prompt
    encoded = model_args.encode_prompt(args.prompt, instruct=args.instruct)
    print(f"\nPrompt : {args.prompt!r}")
    print(
        f"Tokens : {len(encoded)} prompt + {args.max_new_tokens} new = " f"{len(encoded) + args.max_new_tokens} total"
    )

    # ------------------------------------------------------------------ #
    # Weights + TT model                                                   #
    # ------------------------------------------------------------------ #
    print("\nLoading state dict...")
    state_dict = model_args.load_state_dict()

    # ------------------------------------------------------------------ #
    # TT model (non-paged — required for TurboQuant decode path)          #
    # ------------------------------------------------------------------ #
    dtype = ttnn.bfloat8_b
    print("Loading TT model...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=None,  # non-paged: required for turbo_quant_cache path
    )
    del state_dict  # free CPU memory

    # ------------------------------------------------------------------ #
    # Attach TTNNTurboQuantCache to every attention layer                  #
    # ------------------------------------------------------------------ #
    n_local_kv_heads = model_args.n_kv_heads // model_args.cluster_shape[1]
    print(
        f"\nAttaching TurboQuant {args.bits}-bit cache to {len(tt_model.layers)} layers "
        f"(kv_heads={n_local_kv_heads}, head_dim={model_args.head_dim}, "
        f"max_seq={model_args.max_seq_len})..."
    )

    if args.no_turbo_quant:
        print("  (--no-turbo-quant: using standard BFP8 paged_update_cache path)")
    else:
        for layer in tt_model.layers:
            layer.attention.tq_cache = TTNNTurboQuantCache(
                mesh_device,
                num_layers=1,
                num_kv_heads=n_local_kv_heads,
                head_dim=model_args.head_dim,
                max_seq_len=model_args.max_seq_len,
                bits=args.bits,
            )

    # ------------------------------------------------------------------ #
    # Prepare initial inputs                                               #
    # ------------------------------------------------------------------ #
    batch = 1
    total_steps = min(len(encoded) + args.max_new_tokens, model_args.max_seq_len - 1)

    tokens_torch = torch.tensor([encoded[0]], dtype=torch.int64)  # [B=1]
    pos_torch = torch.tensor([0], dtype=torch.int64)  # [B=1]
    host_inputs_0 = tt_model.prepare_decode_inputs_host(tokens_torch, pos_torch)

    use_trace = not args.no_trace

    # ------------------------------------------------------------------ #
    # Warmup: compile all programs (mandatory before trace)               #
    # ------------------------------------------------------------------ #
    print("\nWarmup (compiling programs)...")
    t_compile = time.perf_counter()
    device_inputs_warmup = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
    tt_out_warmup, _ = tt_model.ttnn_decode_forward(*device_inputs_warmup)
    ttnn.deallocate(tt_out_warmup)
    print(f"  Compile time: {time.perf_counter() - t_compile:.1f}s")

    # ------------------------------------------------------------------ #
    # Capture trace (Phase 1.5-A2)                                        #
    # All decode ops captured once; replayed with updated inputs per step. #
    # Eliminates ~3200 Python→C++ dispatch calls per step.                #
    # ------------------------------------------------------------------ #
    if use_trace:
        print("Capturing decode trace...")
        # Allocate fresh device tensors — these are the persistent trace input slots.
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_trace, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        print("  Trace captured.")
    else:
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
        trace_id = None
        tt_out_trace = None

    # ------------------------------------------------------------------ #
    # Decode loop                                                          #
    # ------------------------------------------------------------------ #
    print("\n=== Decode loop ===")
    all_tokens = list(encoded)
    times = []
    current_token = encoded[0]

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)

    for step in range(total_steps):
        # Update pre-allocated device tensors in-place with current step's inputs.
        tokens_torch = torch.tensor([current_token], dtype=torch.int64)
        pos_torch = torch.tensor([step], dtype=torch.int64)
        host_inputs_step = tt_model.prepare_decode_inputs_host(tokens_torch, pos_torch)
        copy_host_to_device(host_inputs_step, device_tensors=trace_inputs)

        t0 = time.perf_counter()
        if use_trace:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            tt_logits = tt_out_trace
        else:
            tt_logits, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Extract logits (blocking read — syncs device)
        logits = (
            ttnn.to_torch(tt_logits, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[:batch, 0:1, : model_args.vocab_size]
        )
        if not use_trace:
            ttnn.deallocate(tt_logits)

        _, next_tok = sample_host(logits, temperature=0, top_p=0.8)
        next_tok_id = int(next_tok.squeeze().item())

        # Teacher-force within prompt, then generate freely.
        if step + 1 < len(encoded):
            current_token = encoded[step + 1]
        else:
            current_token = next_tok_id
            all_tokens.append(next_tok_id)
            tok_str = tokenizer.decode([next_tok_id])
            print(f"  step {step:3d}: {next_tok_id:7d} → {tok_str!r}  ({elapsed*1000:.0f}ms)")

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    print("\n=== Generated text ===")
    print(tokenizer.decode(all_tokens))

    gen_times = times[len(encoded) :]
    if gen_times:
        avg_ms = sum(gen_times) / len(gen_times) * 1000
        mode_label = "baseline BFP8" if args.no_turbo_quant else f"{args.bits}-bit TurboQuant"
        if use_trace:
            mode_label += " (traced)"
        print(f"\n=== Performance ({mode_label}) ===")
        print(f"  Prompt tokens : {len(encoded)}")
        print(f"  Generated     : {len(gen_times)}")
        print(f"  Avg step time : {avg_ms:.1f} ms/tok  ({1000/avg_ms:.1f} tok/s)")
        print(f"  First step    : {times[0]*1000:.0f} ms  (includes compile)")
        if len(times) > 1:
            warm_avg = sum(times[len(encoded) + 1 :]) / max(1, len(times) - len(encoded) - 1) * 1000
            print(f"  Warm avg      : {warm_avg:.1f} ms/tok  (step 2+)")

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #
    if use_trace:
        ttnn.release_trace(mesh_device, trace_id)

    if not args.no_turbo_quant:
        for layer in tt_model.layers:
            tq = getattr(layer.attention, "tq_cache", None)
            if tq is not None:
                tq.deallocate()

    ttnn.close_mesh_device(mesh_device)
    print("\nDone.")


if __name__ == "__main__":
    main()
