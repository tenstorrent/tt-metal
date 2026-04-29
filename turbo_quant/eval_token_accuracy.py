#!/usr/bin/env python3
"""Token-level accuracy benchmark for TurboQuant vs baseline on Llama-3.1-8B.

Teacher-forces through a reference 1024-token corpus (from tests/reference_outputs/)
and measures top-1 and top-5 accuracy. The reference data contains ground-truth
next-token and top-5 candidates at each position, computed from a trusted baseline.

Usage:
    HF_HUB_OFFLINE=1 TT_NUM_DEVICES=8 \\
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \\
    HF_HOME=/localdev/proj_sw/user_dev/hf_data \\
    TT_CACHE_PATH=/localdev/mtairum/hf/ttnn_cache \\
    PYTHONPATH=/localdev/mtairum/tt-metal \\
    python turbo_quant/eval_token_accuracy.py [--bits 3] [--bfp4-cache | --no-turbo-quant]

Baseline expected (from PERF.md, Llama-3.1-8B accuracy mode):
  N150 single device: Top-1 = 96%, Top-5 = 100%
  T3K 8-device:       Top-1 = 97%, Top-5 = 100%
"""
import argparse
import os
import time

import torch
import ttnn

from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, PrecisionSetting, TensorGroup


def build_parser():
    p = argparse.ArgumentParser(description="TurboQuant token accuracy")
    p.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42, help="TurboQuant rotation seed")
    p.add_argument("--no-turbo-quant", action="store_true", help="Baseline BFP8 KV cache")
    p.add_argument(
        "--tq-rescaled-bfp4",
        "--bfp4-cache",
        dest="tq_rescaled_bfp4",
        action="store_true",
        help="Track B: TQ rotation + BFP4 paged layer_past + standard SDPA decode "
        "(centroid×norm pre-rescaled, stored as BFP4). Halves KV memory vs BFP8 baseline. "
        "(--bfp4-cache kept as alias for backwards compatibility.)",
    )
    p.add_argument(
        "--tq-full-dequant",
        action="store_true",
        help="Track A: paged BFP4 indices + BF16 norms in a custom TQ cache, fused "
        "TQ SDPA kernel does centroid-gather × norm on-the-fly. Highest precision; "
        "needs the fused kernel and the multi-device Tier 2A K=14 split for perf.",
    )
    p.add_argument(
        "--tq-recent-window",
        type=int,
        default=0,
        help="Sliding-window hybrid: keep the most recent W tokens in a BFP8 ring. "
        "Only effective with --tq-full-dequant. 0 = disabled (pure Track A).",
    )
    p.add_argument(
        "--bfp4-baseline", action="store_true", help="Baseline with BFP4 cache (no TQ) — isolate storage vs TQ loss"
    )
    p.add_argument(
        "--tq-num-cores-per-head",
        type=int,
        default=1,
        help="Tier 2A K — split each (B, NQH) work item across K cores. 1 = no split. "
        "14 is the validated T3K perf path. Only the non-hybrid SDPA call honours this; "
        "the hybrid combine forces K=1 internally.",
    )
    p.add_argument("--max-seq-len", type=int, default=1024, help="Must be >= 1024 for full reference")
    p.add_argument(
        "--num-eval-tokens",
        type=int,
        default=None,
        help="Number of positions to evaluate (default: from split_point to end, i.e. 512)",
    )
    return p


def main():
    args = build_parser().parse_args()
    assert args.max_seq_len >= 1024, "Reference has 1024 tokens; need max_seq_len >= 1024"

    num_devices = int(os.environ.get("TT_NUM_DEVICES", 1))
    if num_devices > 1:
        print(f"Setting fabric config (FABRIC_1D) for {num_devices} devices...")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    print(f"Opening mesh device ({mesh_shape})...")
    mesh_device = ttnn.open_mesh_device(mesh_shape)

    # ------------------------------------------------------------------ #
    # Model setup (mirrors eval_e2e.py)                                    #
    # ------------------------------------------------------------------ #
    def make_optimizations(ma):
        opts = DecodersPrecision.accuracy(ma.n_layers, ma.model_name)
        if args.tq_rescaled_bfp4 or args.bfp4_baseline:
            for dec_id, dec_conf in opts.decoder_optimizations.items():
                dec_conf.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BFP4
        return opts

    print("Creating ModelArgs...")
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        optimizations=make_optimizations,
        cache_hf=True,
    )
    tokenizer = model_args.tokenizer

    # ------------------------------------------------------------------ #
    # Load reference data                                                  #
    # ------------------------------------------------------------------ #
    # Reference .refpt files are named by full model name (e.g. Llama-3.1-8B-Instruct.refpt).
    ref_file = f"models/tt_transformers/tests/reference_outputs/{model_args.model_name}.refpt"
    if not os.path.exists(ref_file):
        # Fallback: try base model name (some model families use that)
        ref_file = f"models/tt_transformers/tests/reference_outputs/{model_args.base_model_name}.refpt"
    assert os.path.exists(ref_file), f"Reference file not found: {ref_file}"
    print(f"\nLoading reference data: {ref_file}")
    ref_data = torch.load(ref_file, weights_only=False)
    reference_tokens = ref_data["reference_tokens"][0]  # [1024]
    top5_tokens = ref_data["top5_tokens"]  # [1023, 5]
    num_tokens = reference_tokens.shape[0]
    print(f"  Reference: {num_tokens} tokens, top-5 at each position")

    # Evaluation starts at split_point (half), where "continuation" begins.
    split_point = num_tokens // 2
    num_eval = args.num_eval_tokens if args.num_eval_tokens else (num_tokens - 1 - split_point)
    eval_end = min(split_point + num_eval, num_tokens - 1)
    print(f"  Eval positions: {split_point}..{eval_end-1} ({eval_end - split_point} tokens)")

    # ------------------------------------------------------------------ #
    # Model loading with optional rotation absorption                     #
    # ------------------------------------------------------------------ #
    print("\nLoading state dict...")
    state_dict = model_args.load_state_dict()

    use_tq = not args.no_turbo_quant and not args.bfp4_baseline
    if use_tq:
        from turbo_quant.rotation import generate_rotation_matrix
        from turbo_quant.ttnn_integration import absorb_rotation_into_state_dict

        rotation_cpu = generate_rotation_matrix(model_args.head_dim, seed=args.seed, dtype=torch.float32)
        print("  Absorbing Π into W_v and Π^T into W_o (before model load)...")
        absorb_rotation_into_state_dict(
            state_dict,
            rotation_cpu,
            n_layers=model_args.n_layers,
            n_q_heads=model_args.n_heads,
            n_kv_heads=model_args.n_kv_heads,
            head_dim=model_args.head_dim,
        )

    from models.tt_transformers.tt.common import PagedAttentionConfig

    # Need enough blocks for 1024 tokens at batch=1
    block_size = 32
    tq_max_blocks = max(1024, (args.max_seq_len // block_size) + 32)
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=tq_max_blocks)

    dtype = ttnn.bfloat8_b
    from pathlib import Path

    # Reuse the rotated weight cache that eval_e2e.py creates (suffix "_tq_rotated").
    # If absent it'll be regenerated on first run; previously this used a tempdir
    # which wastes disk and forces full re-tilization on every accuracy run.
    wcache = model_args.weight_cache_path(dtype)
    if use_tq:
        wcache = Path(str(wcache) + "_tq_rotated")

    print("Loading TT model...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=wcache,
        paged_attention_config=paged_attention_config,
    )
    del state_dict

    # ------------------------------------------------------------------ #
    # Attach TQ cache (for non-baseline modes)                            #
    # ------------------------------------------------------------------ #
    if use_tq:
        from turbo_quant.ttnn_integration import TTNNTurboQuantCache

        # IMPORTANT: One shared TQ cache for all layers, not 32 separate caches.
        # Per-layer caches caused decode to hang at scale (each layer compiles its
        # own program and the program cache misses dominate).
        n_local_kv_heads = model_args.n_kv_heads // model_args.cluster_shape[1]
        if args.tq_full_dequant:
            # Track A: free model's standard layer_past (fused path owns the KV cache)
            # and allocate a paged BFP4-idx + BF16-norms TQ cache.
            for layer in tt_model.layers:
                attn = layer.attention
                if hasattr(attn, "layer_past") and attn.layer_past is not None:
                    for t in attn.layer_past:
                        ttnn.deallocate(t)
                    attn.layer_past = None
            shared_tq = TTNNTurboQuantCache(
                mesh_device,
                num_layers=len(tt_model.layers),
                num_kv_heads=n_local_kv_heads,
                head_dim=model_args.head_dim,
                max_seq_len=args.max_seq_len,
                bits=args.bits,
                memory_efficient=True,  # paged BFP4 indices + BF16 norms + fused SDPA
                paged_config=paged_attention_config,
                max_batch_size=1,
                seed=args.seed,
                recent_window=args.tq_recent_window,
            )
        else:
            # Track B / pre-rescaled mode: TQ object only carries the rotation matrix
            # and codebook; the actual storage is the model's standard layer_past.
            shared_tq = TTNNTurboQuantCache(
                mesh_device,
                num_layers=1,
                num_kv_heads=n_local_kv_heads,
                head_dim=model_args.head_dim,
                max_seq_len=32,
                bits=args.bits,
                memory_efficient=False,
                seed=args.seed,
            )
        shared_tq.rotation_absorbed = True
        for layer_idx, layer in enumerate(tt_model.layers):
            layer.attention.tq_cache = shared_tq
            if args.tq_full_dequant:
                layer.attention.tq_layer_idx = layer_idx

        # Tell the model to build ring_pos / old_pos device tensors in
        # prepare_decode_inputs_host (replaces the per-step host sync that
        # used to break trace).
        if args.tq_recent_window > 0:
            tt_model.tq_ring_W = shared_tq.recent_window
            tt_model.tq_ring_W_padded = shared_tq.ring_W_padded

        # Tier 2A K split (only honoured by the non-hybrid SDPA path).
        shared_tq.num_cores_per_head = args.tq_num_cores_per_head

    # ------------------------------------------------------------------ #
    # Teacher-forced decode: feed ref tokens one at a time, get top-5    #
    # ------------------------------------------------------------------ #
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)
    page_table_cpu = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).unsqueeze(0)

    # Warmup + trace capture. The sliding-window path now ships ring_pos /
    # old_pos as trace-input device tensors, so trace is fully compatible.
    use_trace = not bool(os.environ.get("TQ_DUMP_LSE", ""))
    print("\nWarmup (compile decode programs)...")
    tokens_torch = torch.tensor([int(reference_tokens[0].item())], dtype=torch.int64)
    pos_torch = torch.tensor([0], dtype=torch.int64)
    host_inputs_0 = tt_model.prepare_decode_inputs_host(tokens_torch, pos_torch, page_table=page_table_cpu)
    t_compile = time.perf_counter()
    device_inputs_w = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
    tt_out_w, _ = tt_model.ttnn_decode_forward(*device_inputs_w)
    ttnn.deallocate(tt_out_w)
    print(f"  Compile time: {time.perf_counter() - t_compile:.1f}s")

    if use_trace:
        print("Capturing trace...")
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_trace, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        print("  Trace captured.")
    else:
        trace_id = None
        trace_inputs = None
        tt_out_trace = None

    # Teacher-force through all 1023 positions, record top-5 at each.
    print(f"\nRunning teacher-forced decode over {num_tokens - 1} positions...")
    top1_correct = 0
    top5_correct = 0
    total_eval = 0
    times = []
    t0_total = time.perf_counter()

    for pos in range(num_tokens - 1):
        # Input: ref token at this position. Output: predicted next token.
        tok = int(reference_tokens[pos].item())
        tokens_torch = torch.tensor([tok], dtype=torch.int64)
        pos_torch = torch.tensor([pos], dtype=torch.int64)
        host_inputs_step = tt_model.prepare_decode_inputs_host(tokens_torch, pos_torch, page_table=page_table_cpu)

        if use_trace:
            copy_host_to_device(host_inputs_step, device_tensors=trace_inputs)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            times.append(time.perf_counter() - t0)
            out_tensor = tt_out_trace
        else:
            device_inputs_step = copy_host_to_device(host_inputs_step, mesh_device=mesh_device)
            t0 = time.perf_counter()
            out_tensor, _ = tt_model.ttnn_decode_forward(*device_inputs_step)
            times.append(time.perf_counter() - t0)

        # Extract logits [1, 1, batch*tile, vocab] → [batch, 1, vocab]
        logits = (
            ttnn.to_torch(out_tensor, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[:1, 0:1, : model_args.vocab_size]
        )
        if not use_trace:
            ttnn.deallocate(out_tensor)

        # Top-5 predicted tokens at this position.
        logits_flat = logits.squeeze()  # [vocab]
        top5_pred = torch.topk(logits_flat, 5).indices.tolist()

        # Only evaluate positions in the "continuation" half.
        # Compare model's top-1 prediction vs top5_tokens[pos, 0] — which is the
        # reference model's greedy top-1 at this position. Also check top-5.
        if pos >= split_point - 1 and pos < eval_end:
            ref_top5 = top5_tokens[pos, :].tolist()
            ref_top1 = ref_top5[0]
            if top5_pred[0] == ref_top1:
                top1_correct += 1
            if top5_pred[0] in ref_top5:
                top5_correct += 1
            total_eval += 1

            if total_eval % 50 == 0:
                print(
                    f"  pos={pos}: top1={top1_correct}/{total_eval} ({100*top1_correct/total_eval:.1f}%)  "
                    f"top5={top5_correct}/{total_eval} ({100*top5_correct/total_eval:.1f}%)"
                )

    elapsed_total = time.perf_counter() - t0_total
    top1_pct = 100 * top1_correct / total_eval if total_eval else 0
    top5_pct = 100 * top5_correct / total_eval if total_eval else 0

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    if args.no_turbo_quant:
        mode = "baseline BFP8"
    elif args.bfp4_baseline:
        mode = "baseline BFP4 (no TQ)"
    elif args.tq_rescaled_bfp4:
        mode = f"TQ {args.bits}-bit rescaled BFP4 paged"
    elif args.tq_full_dequant:
        mode = f"TQ {args.bits}-bit full dequant (fused kernel)"
    else:
        mode = f"TurboQuant {args.bits}-bit BF16"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode} ({num_devices}-device)")
    print(f"{'='*60}")
    print(f"  Top-1 accuracy: {top1_correct}/{total_eval} = {top1_pct:.1f}%")
    print(f"  Top-5 accuracy: {top5_correct}/{total_eval} = {top5_pct:.1f}%")
    warm_times = times[10:]  # skip first few warmup-ish steps
    if warm_times:
        print(f"  Avg latency: {1000 * sum(warm_times)/len(warm_times):.1f} ms/tok")
    print(f"  Total eval time: {elapsed_total:.1f}s")
    print(f"{'='*60}")

    if use_trace:
        ttnn.release_trace(mesh_device, trace_id)
    if use_tq:
        # Single shared TQ cache — only deallocate once (all layers point to the same one).
        shared_tq.deallocate()
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
