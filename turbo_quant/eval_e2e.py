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
    p.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--instruct", action="store_true", default=True, help="Use instruct prompt formatting")
    p.add_argument("--num-layers", type=int, default=None, help="Limit to N layers (default: all 32)")
    p.add_argument("--no-turbo-quant", action="store_true", help="Baseline: skip TurboQuant, use standard BFP8 cache")
    p.add_argument("--bfp4-cache", action="store_true", help="Use BFP4 paged cache (0.5 bytes/elem) instead of BF16")
    p.add_argument(
        "--tq-full-dequant",
        action="store_true",
        help="Use TQ Full Dequant path: paged BFP4 indices + BF16 norms cache + fused SDPA "
        "(centroid gather × norm on-the-fly inside SDPA kernel). "
        "Expected to preserve accuracy (>95%% top-1) while keeping 2x memory savings vs baseline.",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (number of parallel sequences)")
    p.add_argument("--no-trace", action="store_true", help="Disable TTNN trace (slower, useful for debugging)")
    return p


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ #
    # Device + model setup                                                 #
    # ------------------------------------------------------------------ #
    # Single device: MeshShape(1, 1). T3K: MeshShape(1, 8) + FABRIC_1D.
    import os

    num_devices = int(os.environ.get("TT_NUM_DEVICES", 1))
    if num_devices > 1:
        # Multi-device needs fabric for inter-device communication (paged attention,
        # all-gather, all-reduce). T3K uses 1D fabric.
        print(f"Setting fabric config (FABRIC_1D) for {num_devices} devices...")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    print(f"Opening mesh device ({mesh_shape})...")
    mesh_device = ttnn.open_mesh_device(mesh_shape)

    print(f"Creating ModelArgs (batch_size={args.batch_size})...")

    # For BFP4 cache mode, override KV_CACHE precision to BFP4 in the default
    # accuracy optimization. Otherwise model init allocates BFP8 paged cache,
    # limiting max_num_blocks to BFP8's memory footprint (defeating TQ's 2x
    # memory savings at max-batch stress tests).
    def make_optimizations(ma):
        from models.tt_transformers.tt.model_config import TensorGroup, PrecisionSetting

        opts = DecodersPrecision.accuracy(ma.n_layers, ma.model_name)
        if args.bfp4_cache:
            for decoder_id, dec_conf in opts.decoder_optimizations.items():
                dec_conf.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BFP4
        return opts

    model_args = ModelArgs(
        mesh_device,
        instruct=args.instruct,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        optimizations=make_optimizations,
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
    # Absorb rotation into weights (before model creation)                 #
    # ------------------------------------------------------------------ #
    absorb_rotation = not args.no_turbo_quant
    if absorb_rotation:
        from turbo_quant.rotation import generate_rotation_matrix
        from turbo_quant.ttnn_integration import absorb_rotation_into_state_dict

        rotation_cpu = generate_rotation_matrix(model_args.head_dim, seed=42, dtype=torch.float32)
        print("  Absorbing Π into W_v and Π^T into W_o (before model load)...")
        absorb_rotation_into_state_dict(
            state_dict,
            rotation_cpu,
            n_layers=model_args.n_layers,
            n_q_heads=model_args.n_heads,
            n_kv_heads=model_args.n_kv_heads,
            head_dim=model_args.head_dim,
        )

    # ------------------------------------------------------------------ #
    # TT model                                                              #
    # ------------------------------------------------------------------ #
    dtype = ttnn.bfloat8_b
    wcache = model_args.weight_cache_path(dtype)
    if absorb_rotation:
        from pathlib import Path

        wcache = Path(str(wcache) + "_tq_rotated")

    # Paged attention: same model config as baseline.  TQ uses its own BF16
    # contiguous cache + non-paged SDPA; the model's paged layer_past is
    # allocated but bypassed when tq_cache is active.
    from models.tt_transformers.tt.common import PagedAttentionConfig

    # Block budget must cover batch_size × max_seq_len tokens.
    # Each batch gets its own contiguous slab of pages.
    block_size = 32
    blocks_per_batch = (args.max_seq_len + block_size - 1) // block_size
    min_blocks = args.batch_size * blocks_per_batch
    # Keep at least 1024 blocks to match existing warmup/trace state.
    tq_max_blocks = max(1024, min_blocks + 32)  # +32 padding for tile alignment
    print(
        f"  Paged attention: block_size={block_size}, max_num_blocks={tq_max_blocks} "
        f"(batch={args.batch_size} x {blocks_per_batch} blocks/batch)"
    )
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=tq_max_blocks)
    print("Loading TT model (paged attention, block_size=32)...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=wcache,
        paged_attention_config=paged_attention_config,
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
    elif args.tq_full_dequant:
        # Full Dequant: TQ cache stores paged BFP4 indices + BF16 norms separately.
        # Fused SDPA decode reads both, reconstructs centroid×norm on-the-fly inside
        # the kernel, and feeds into chunked online softmax.
        # Free the model's BFP8 layer_past to reclaim DRAM (fused path doesn't use it).
        print(
            f"  Full Dequant path: allocating paged TQ cache "
            f"(blocks={paged_attention_config.max_num_blocks}, block_size={paged_attention_config.block_size})"
        )
        print("  Freeing model's BFP8 layer_past (fused path owns the KV cache)...")
        for layer in tt_model.layers:
            attn = layer.attention
            if hasattr(attn, "layer_past") and attn.layer_past is not None:
                for t in attn.layer_past:
                    ttnn.deallocate(t)
                attn.layer_past = None

        # Single shared cache with one slot per layer — reuses the same kernel
        # program across all 32 layers (different layer_idx → different cache slot
        # but same program signature → program cache hits). Creating 32 separate
        # num_layers=1 caches caused the decode to hang at scale.
        shared_tq = TTNNTurboQuantCache(
            mesh_device,
            num_layers=len(tt_model.layers),
            num_kv_heads=n_local_kv_heads,
            head_dim=model_args.head_dim,
            max_seq_len=model_args.max_seq_len,
            bits=args.bits,
            memory_efficient=True,  # paged BFP4 indices + BF16 norms
            paged_config=paged_attention_config,
            max_batch_size=args.batch_size,
        )
        if absorb_rotation:
            shared_tq.rotation_absorbed = True
        for layer_idx, layer in enumerate(tt_model.layers):
            attn = layer.attention
            attn.tq_cache = shared_tq
            attn.tq_layer_idx = layer_idx
    elif args.bfp4_cache:
        # KV_CACHE already allocated as BFP4 at model init (via make_optimizations override).
        print("  Using BFP4 paged cache (allocated at model init via optimization override)")
    else:
        # BF16 mode: model init'd BFP8, swap to BF16 (2 bytes/elem).
        print("  Replacing BFP8 layer_past with BF16 paged cache...")
        lp_shape = None
        for layer in tt_model.layers:
            attn = layer.attention
            if hasattr(attn, "layer_past") and attn.layer_past is not None:
                lp_shape = list(attn.layer_past[0].shape)
                for t in attn.layer_past:
                    ttnn.deallocate(t)
                attn.layer_past = None

        for layer in tt_model.layers:
            layer.attention.layer_past = [
                ttnn.from_torch(
                    torch.zeros(lp_shape, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)  # K, V
            ]

            # Attach TQ setup (rotation matrix + codebook only, no separate cache).
            tq = TTNNTurboQuantCache(
                mesh_device,
                num_layers=1,
                num_kv_heads=n_local_kv_heads,
                head_dim=model_args.head_dim,
                max_seq_len=32,  # minimal — TQ no longer manages its own cache
                bits=args.bits,
                memory_efficient=False,  # pre-rescale for BF16 paged cache
            )
            if absorb_rotation:
                tq.rotation_absorbed = True
            attn.tq_cache = tq

    # ------------------------------------------------------------------ #
    # Prepare initial inputs                                               #
    # ------------------------------------------------------------------ #
    batch = args.batch_size
    total_steps = min(len(encoded) + args.max_new_tokens, model_args.max_seq_len - 1)

    # Page table: identity mapping per batch slot. Shape [batch, max_num_blocks].
    # Each batch gets distinct pages (interleaved): batch b uses pages [b, b+batch, b+2*batch, ...]
    # Simple approach: give each batch its own contiguous block of pages.
    pages_per_batch = paged_attention_config.max_num_blocks // batch
    page_table_cpu = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32)
    page_table_cpu = page_table_cpu[: batch * pages_per_batch].reshape(batch, pages_per_batch)

    # Replicate same prompt across all batch slots for throughput testing.
    tokens_torch = torch.tensor([encoded[0]] * batch, dtype=torch.int64)  # [B]
    pos_torch = torch.tensor([0] * batch, dtype=torch.int64)  # [B]
    host_inputs_0 = tt_model.prepare_decode_inputs_host(tokens_torch, pos_torch, page_table=page_table_cpu)

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
    current_tokens = [encoded[0]] * batch  # one token per batch slot

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)

    for step in range(total_steps):
        # Update pre-allocated device tensors in-place with current step's inputs.
        tokens_torch = torch.tensor(current_tokens, dtype=torch.int64)
        pos_torch = torch.tensor([step] * batch, dtype=torch.int64)
        host_inputs_step = tt_model.prepare_decode_inputs_host(tokens_torch, pos_torch, page_table=page_table_cpu)
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
        # next_tok shape: [batch, 1]. Extract token id per batch slot.
        next_tok_ids = [int(next_tok[b].squeeze().item()) for b in range(batch)]

        # Teacher-force within prompt, then generate freely. All batch slots
        # get the same prompt, so they produce the same tokens.
        if step + 1 < len(encoded):
            current_tokens = [encoded[step + 1]] * batch
        else:
            current_tokens = next_tok_ids
            all_tokens.append(next_tok_ids[0])  # only log first batch slot
            tok_str = tokenizer.decode([next_tok_ids[0]])
            print(f"  step {step:3d}: {next_tok_ids[0]:7d} → {tok_str!r}  ({elapsed*1000:.0f}ms)")

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    print("\n=== Generated text ===")
    print(tokenizer.decode(all_tokens))

    gen_times = times[len(encoded) :]
    if gen_times:
        avg_ms = sum(gen_times) / len(gen_times) * 1000
        if args.no_turbo_quant:
            mode_label = "baseline BFP8"
        elif args.bfp4_cache:
            mode_label = f"{args.bits}-bit TurboQuant BFP4 paged"
        else:
            mode_label = f"{args.bits}-bit TurboQuant BF16 paged"
        if use_trace:
            mode_label += " (traced)"
        print(f"\n=== Performance ({mode_label}, batch={batch}) ===")
        print(f"  Prompt tokens : {len(encoded)}")
        print(f"  Generated     : {len(gen_times)}")
        print(f"  Avg step time : {avg_ms:.1f} ms/tok  ({1000/avg_ms:.1f} tok/s single-seq)")
        if batch > 1:
            throughput = batch * 1000 / avg_ms
            print(f"  Throughput    : {throughput:.1f} tok/s (batch={batch})")
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
