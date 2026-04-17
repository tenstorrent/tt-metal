#!/usr/bin/env python3
"""Compare TurboQuant vs baseline BFP8 output quality on the ci-eval-1 prompts.

Runs each prompt through:
  1. Baseline BFP8 KV cache (standard Llama path)
  2. TurboQuant 3-bit with fused kernels + rotated SDPA + BFP8 indices

Reports: generated text, token count, ms/tok for both paths.

Usage:
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    HF_HOME=/localdev/proj_sw/user_dev/hf_data \
    TT_CACHE_PATH=/localdev/proj_sw/user_dev/hf_data/ttnn_cache \
    PYTHONPATH=/localdev/mtairum/tt-metal \
    python turbo_quant/eval_quality_comparison.py [--bits 3] [--max-new-tokens 200]
"""

import argparse
import json
import time

import torch
import ttnn

from models.tt_transformers.tt.common import Mode, copy_host_to_device, sample_host
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from turbo_quant.ttnn_integration import TTNNTurboQuantCache
from turbo_quant.quantizer import TurboQuantMSE


def build_parser():
    p = argparse.ArgumentParser(description="TurboQuant vs baseline quality comparison")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument(
        "--prompts-file",
        default="models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch1.json",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def load_prompts(path):
    """Load prompts, deduplicate (the file has pairs of identical prompts)."""
    with open(path) as f:
        data = json.load(f)
    seen = set()
    unique = []
    for entry in data:
        p = entry["prompt"]
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def generate_baseline(tt_model, model_args, mesh_device, prompt_text, max_new_tokens, max_seq_len):
    """Generate with standard BFP8 KV cache (no TurboQuant)."""
    tokenizer = model_args.tokenizer
    encoded = model_args.encode_prompt(prompt_text, instruct=True)
    prompt_len = len(encoded)
    if prompt_len >= max_seq_len:
        return "(prompt too long)", 0, 0.0

    # Prefill
    tokens_2d = torch.tensor([encoded])
    pad_to = ((prompt_len + 127) // 128) * 128
    if pad_to > prompt_len:
        tokens_2d = torch.cat([tokens_2d, torch.zeros(1, pad_to - prompt_len, dtype=torch.long)], dim=1)

    get_last_token = ((prompt_len - 1) // 32) * 32
    (
        prefill_input,
        rot_mats_global,
        rot_mats_local,
        tt_page_table,
        tt_chunk_page_table,
    ) = tt_model.prepare_inputs_prefill(tokens_2d)
    tt_prefill_out = tt_model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        page_table=tt_page_table,
        get_last_token=get_last_token,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)
    prefill_logits = (
        ttnn.to_torch(tt_prefill_out, mesh_composer=mesh_composer)
        .permute(2, 1, 0, 3)
        .squeeze(2)[:, 0:1, : model_args.vocab_size]
    )
    ttnn.deallocate(tt_prefill_out)

    last_row = (prompt_len - 1) - get_last_token
    logits_last = prefill_logits[last_row : last_row + 1, :, :]
    _, next_tok = sample_host(logits_last, temperature=0, top_p=0.8)
    current_tok_id = int(next_tok.squeeze().item())

    eot_id = tokenizer.eos_token_id
    all_new_tokens = [current_tok_id]
    times = []

    for step in range(max_new_tokens - 1):
        pos = prompt_len + step
        host_inputs = tt_model.prepare_decode_inputs_host(
            torch.tensor([current_tok_id], dtype=torch.int64),
            torch.tensor([pos], dtype=torch.int64),
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)

        t0 = time.perf_counter()
        tt_logits, _ = tt_model.ttnn_decode_forward(*device_inputs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        logits = (
            ttnn.to_torch(tt_logits, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[:1, 0:1, : model_args.vocab_size]
        )
        ttnn.deallocate(tt_logits)

        _, next_tok = sample_host(logits, temperature=0, top_p=0.8)
        current_tok_id = int(next_tok.squeeze().item())
        all_new_tokens.append(current_tok_id)

        if current_tok_id == eot_id:
            break

    output_text = tokenizer.decode(all_new_tokens)
    avg_ms = sum(times) / len(times) * 1000 if times else 0
    return output_text, len(all_new_tokens), avg_ms


def generate_turbo_quant(tt_model, model_args, mesh_device, prompt_text, max_new_tokens, max_seq_len, bits, seed):
    """Generate with TurboQuant BFP4 KV cache."""
    tokenizer = model_args.tokenizer
    encoded = model_args.encode_prompt(prompt_text, instruct=True)
    prompt_len = len(encoded)
    if prompt_len >= max_seq_len:
        return "(prompt too long)", 0, 0.0

    # Prefill (same as baseline — fills layer_past with BFP8)
    tokens_2d = torch.tensor([encoded])
    pad_to = ((prompt_len + 127) // 128) * 128
    if pad_to > prompt_len:
        tokens_2d = torch.cat([tokens_2d, torch.zeros(1, pad_to - prompt_len, dtype=torch.long)], dim=1)

    get_last_token = ((prompt_len - 1) // 32) * 32
    (
        prefill_input,
        rot_mats_global,
        rot_mats_local,
        tt_page_table,
        tt_chunk_page_table,
    ) = tt_model.prepare_inputs_prefill(tokens_2d)
    tt_prefill_out = tt_model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        page_table=tt_page_table,
        get_last_token=get_last_token,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)
    prefill_logits = (
        ttnn.to_torch(tt_prefill_out, mesh_composer=mesh_composer)
        .permute(2, 1, 0, 3)
        .squeeze(2)[:, 0:1, : model_args.vocab_size]
    )
    ttnn.deallocate(tt_prefill_out)

    last_row = (prompt_len - 1) - get_last_token
    logits_last = prefill_logits[last_row : last_row + 1, :, :]
    _, next_tok = sample_host(logits_last, temperature=0, top_p=0.8)
    current_tok_id = int(next_tok.squeeze().item())

    # Migrate prefill KV to TQ pre-rescaled format.
    # Non-paged update_cache only supports BF16/BFP8 (not BFP4), so use BF16 here.
    # The paged path (eval_e2e.py --bfp4-cache) uses BFP4 via paged_update_cache.
    from turbo_quant.eval_e2e_prefill import migrate_prefill_kv_to_turbo_quant

    migrate_prefill_kv_to_turbo_quant(
        tt_model,
        prompt_len=prompt_len,
        bits=bits,
        mesh_device=mesh_device,
        seed=seed,
        rotation_absorbed=True,
        cache_dtype=ttnn.bfloat16,
        cluster_shape=model_args.cluster_shape,
    )

    # Attach TQ cache for decode quantize path (not for SDPA — standard SDPA reads BFP4)
    n_local_kv_heads = model_args.n_kv_heads // model_args.cluster_shape[1]
    for layer in tt_model.layers:
        tq = TTNNTurboQuantCache(
            mesh_device,
            num_layers=1,
            num_kv_heads=n_local_kv_heads,
            head_dim=model_args.head_dim,
            max_seq_len=max_seq_len,
            bits=bits,
            seed=seed,
            memory_efficient=False,
        )
        tq.rotation_absorbed = True
        layer.attention.tq_cache = tq

    eot_id = tokenizer.eos_token_id
    all_new_tokens = [current_tok_id]
    times = []

    for step in range(max_new_tokens - 1):
        pos = prompt_len + step
        host_inputs = tt_model.prepare_decode_inputs_host(
            torch.tensor([current_tok_id], dtype=torch.int64),
            torch.tensor([pos], dtype=torch.int64),
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)

        t0 = time.perf_counter()
        tt_logits, _ = tt_model.ttnn_decode_forward(*device_inputs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        logits = (
            ttnn.to_torch(tt_logits, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[:1, 0:1, : model_args.vocab_size]
        )
        ttnn.deallocate(tt_logits)

        _, next_tok = sample_host(logits, temperature=0, top_p=0.8)
        current_tok_id = int(next_tok.squeeze().item())
        all_new_tokens.append(current_tok_id)

        if current_tok_id == eot_id:
            break

    # Cleanup TQ caches
    for layer in tt_model.layers:
        tq = getattr(layer.attention, "tq_cache", None)
        if tq is not None:
            tq.deallocate()
            delattr(layer.attention, "tq_cache")

    output_text = tokenizer.decode(all_new_tokens)
    avg_ms = sum(times) / len(times) * 1000 if times else 0
    return output_text, len(all_new_tokens), avg_ms


def main():
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts_file)

    print(f"Loaded {len(prompts)} unique prompts from {args.prompts_file}")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")

    import os

    num_devices = int(os.environ.get("TT_NUM_DEVICES", 1))
    if num_devices > 1:
        print(f"Setting fabric config (FABRIC_1D) for {num_devices} devices...")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    print(f"Opening mesh device ({mesh_shape})...")
    mesh_device = ttnn.open_mesh_device(mesh_shape)

    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
        cache_hf=True,
    )
    tokenizer = model_args.tokenizer
    dtype = ttnn.bfloat8_b

    # ------------------------------------------------------------------ #
    # Phase 1: Baseline BFP8 (no rotation absorption)                     #
    # ------------------------------------------------------------------ #
    print("\n=== Loading baseline model (no rotation) ===")
    state_dict = model_args.load_state_dict()
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=None,
    )
    del state_dict

    baseline_results = {}
    for i, prompt in enumerate(prompts):
        short = prompt[:60] + ("..." if len(prompt) > 60 else "")
        print(f"\n--- Baseline BFP8 | Prompt {i}: {short} ---")
        text, toks, ms = generate_baseline(
            tt_model, model_args, mesh_device, prompt, args.max_new_tokens, args.max_seq_len
        )
        print(f"  Tokens: {toks}, Avg: {ms:.1f} ms/tok")
        print(f"  Output: {text[:300]}{'...' if len(text) > 300 else ''}")
        baseline_results[i] = {"text": text, "tokens": toks, "ms": ms}

    # Free baseline model
    del tt_model
    import gc

    gc.collect()

    # ------------------------------------------------------------------ #
    # Phase 2: TurboQuant BFP4 (rotation-absorbed weights)               #
    # ------------------------------------------------------------------ #
    print("\n=== Loading TQ model (rotation-absorbed) ===")
    state_dict = model_args.load_state_dict()

    from turbo_quant.rotation import generate_rotation_matrix
    from turbo_quant.ttnn_integration import absorb_rotation_into_state_dict

    rotation_cpu = generate_rotation_matrix(model_args.head_dim, seed=args.seed, dtype=torch.float32)
    print("  Absorbing Π into W_v and Π^T into W_o...")
    absorb_rotation_into_state_dict(
        state_dict,
        rotation_cpu,
        n_layers=model_args.n_layers,
        n_q_heads=model_args.n_heads,
        n_kv_heads=model_args.n_kv_heads,
        head_dim=model_args.head_dim,
    )

    from pathlib import Path
    import tempfile

    wcache = Path(tempfile.mkdtemp(prefix="tq_quality_weights_"))
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=wcache,
        paged_attention_config=None,
    )
    del state_dict

    tq_results = {}
    for i, prompt in enumerate(prompts):
        short = prompt[:60] + ("..." if len(prompt) > 60 else "")
        print(f"\n--- TurboQuant {args.bits}-bit BFP4 | Prompt {i}: {short} ---")
        text, toks, ms = generate_turbo_quant(
            tt_model, model_args, mesh_device, prompt, args.max_new_tokens, args.max_seq_len, args.bits, args.seed
        )
        print(f"  Tokens: {toks}, Avg: {ms:.1f} ms/tok")
        print(f"  Output: {text[:300]}{'...' if len(text) > 300 else ''}")
        tq_results[i] = {"text": text, "tokens": toks, "ms": ms}

    del tt_model

    # ------------------------------------------------------------------ #
    # Phase 3: Comparison                                                  #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    for i, prompt in enumerate(prompts):
        short = prompt[:60] + ("..." if len(prompt) > 60 else "")
        base = baseline_results[i]
        tq = tq_results[i]
        match = base["text"].strip() == tq["text"].strip()

        print(f"\nPrompt {i}: {short}")
        print(f"  Baseline: {base['tokens']} toks, {base['ms']:.1f} ms/tok")
        print(f"  TQ BFP4:  {tq['tokens']} toks, {tq['ms']:.1f} ms/tok")
        print(f"  Exact match: {'YES' if match else 'NO'}")

        if not match and base["tokens"] > 0 and tq["tokens"] > 0:
            base_words = base["text"].split()
            tq_words = tq["text"].split()
            common = len(set(base_words) & set(tq_words))
            total = len(set(base_words) | set(tq_words))
            if total > 0:
                print(f"  Word overlap: {common}/{total} ({100*common/total:.0f}%)")
            print(f"  Baseline output: {base['text'][:200]}")
            print(f"  TQ BFP4 output:  {tq['text'][:200]}")

    ttnn.close_mesh_device(mesh_device)
    print("\nDone.")


if __name__ == "__main__":
    main()
