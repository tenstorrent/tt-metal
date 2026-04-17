#!/usr/bin/env python3
"""Llama-3.1-8B-Instruct inference with proper prefill + TurboQuant KV cache decode.

Compared to eval_e2e.py (teacher-forced), this script runs a real single-pass
prefill for all prompt tokens, migrates the resulting BFP8 KV cache into
TurboQuant CPU shadow buffers, and then generates freely token-by-token.

Flow:
  1. Prefill  — forward pass over full prompt (all tokens at once); fills layer_past.
  2. Migrate  — cast layer_past BFP8 → BF16 → CPU float32, quantize with TurboQuantMSE,
                store indices + norms in tq_cache CPU shadow for positions [0..prompt_len-1].
  3. Decode   — standard TurboQuant decode from position prompt_len; stops on <|eot_id|>
                or max_new_tokens.

Usage:
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \\
    HF_HOME=/localdev/mtairum/hf \\
    TT_CACHE_PATH=/localdev/mtairum/hf/ttnn_cache \\
    PYTHONPATH=/localdev/mtairum/tt-metal \\
    python turbo_quant/eval_e2e_prefill.py [--prompt "..."] [--bits 3] [--max-new-tokens 60]
"""

import argparse
import time

import torch
import ttnn

from models.tt_transformers.tt.common import Mode, copy_host_to_device, sample_host
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from turbo_quant.quantizer import TurboQuantMSE
from turbo_quant.ttnn_integration import TTNNTurboQuantCache


def build_parser():
    p = argparse.ArgumentParser(description="TurboQuant prefill + decode demo")
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--instruct", action="store_true", default=True)
    p.add_argument("--num-layers", type=int, default=None, help="Limit to N layers (debug)")
    p.add_argument("--seed", type=int, default=42, help="TurboQuant rotation seed")
    p.add_argument("--bfp4-cache", action="store_true", help="Use BFP4 paged cache (0.5 bytes/elem)")
    p.add_argument("--no-trace", action="store_true", help="Disable TTNN trace (slower, useful for debugging)")
    return p


def migrate_prefill_kv_to_turbo_quant(
    tt_model,
    prompt_len: int,
    bits: int,
    mesh_device,
    seed: int = 42,
    rotation_absorbed: bool = False,
    cache_dtype=None,
    cluster_shape=None,
    paged: bool = False,
    block_size: int = 32,
):
    """Migrate prefill KV from layer_past into TurboQuant pre-rescaled format.

    After prefill, each layer's layer_past holds BFP8 KV values. This function:
      1. Reads each layer_past slab → CPU float32.
      2. Applies TurboQuant rotation + quantization → centroid × norm values.
      3. Writes pre-rescaled values back into layer_past.

    Args:
        tt_model: Loaded Transformer with layer_past filled by prefill.
        prompt_len: Number of prompt tokens (valid positions in layer_past).
        bits: Quantisation bit-width.
        mesh_device: TTNN mesh device.
        seed: Rotation matrix seed (must match TQ setup).
        rotation_absorbed: If True, V from prefill is already in rotated space.
        cache_dtype: Target ttnn dtype for layer_past (default: bfloat16).
    """
    if cache_dtype is None:
        cache_dtype = ttnn.bfloat16
    head_dim = tt_model.layers[0].attention.head_dim
    cpu_quantizer = TurboQuantMSE(
        head_dim=head_dim,
        bits=bits,
        seed=seed,
        device="cpu",
        dtype=torch.float32,
    )

    dtype_label = {ttnn.bfloat16: "BF16", ttnn.bfloat4_b: "BFP4", ttnn.bfloat8_b: "BFP8"}.get(
        cache_dtype, str(cache_dtype)
    )

    # Multi-device: each device holds n_local_kv_heads. Concat along dim 1 (heads)
    # when reading back, shard along dim 1 when writing back.
    num_devices = cluster_shape[1] if cluster_shape else 1
    if num_devices > 1:
        mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=cluster_shape)
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=cluster_shape)
    else:
        mesh_composer = None
        mesh_mapper = None

    print(
        f"  Migrating {len(tt_model.layers)} layers × {prompt_len} positions to TQ {dtype_label} pre-rescaled (paged={paged}) ..."
    )
    for layer in tt_model.layers:
        # Read BFP8 KV → CPU float32.
        # Paged layout:  [max_num_blocks, n_kv_heads, block_size, head_dim]  (block-major)
        # Flat layout:   [batch, n_kv_heads, max_seq, head_dim]
        k_bf16 = ttnn.typecast(layer.attention.layer_past[0], ttnn.bfloat16)
        v_bf16 = ttnn.typecast(layer.attention.layer_past[1], ttnn.bfloat16)
        k_cpu = ttnn.to_torch(k_bf16, mesh_composer=mesh_composer).float()
        v_cpu = ttnn.to_torch(v_bf16, mesh_composer=mesh_composer).float()
        ttnn.deallocate(k_bf16)
        ttnn.deallocate(v_bf16)

        orig_shape = list(k_cpu.shape)
        if paged:
            # [max_num_blocks, n_kv_heads, block_size, head_dim]
            #   → [n_kv_heads, max_num_blocks * block_size, head_dim]
            #   → [1, n_kv_heads, max_seq, head_dim]
            max_num_blocks, n_kv_heads, blk, _ = orig_shape
            max_seq = max_num_blocks * blk
            k_cpu = k_cpu.permute(1, 0, 2, 3).reshape(n_kv_heads, max_seq, head_dim).unsqueeze(0)
            v_cpu = v_cpu.permute(1, 0, 2, 3).reshape(n_kv_heads, max_seq, head_dim).unsqueeze(0)
        else:
            max_seq = k_cpu.shape[2]

        k_prefix = k_cpu[:, :, :prompt_len, :]
        v_prefix = v_cpu[:, :, :prompt_len, :]

        # K: has RoPE but NOT rotated by Π → explicit rotation needed.
        k_idx, k_norms = cpu_quantizer.quantize(k_prefix)
        k_centroids = cpu_quantizer.codebook.centroids[k_idx.long()]
        k_rescaled = (k_centroids * k_norms).to(torch.bfloat16)

        # V: already in rotated space if rotation_absorbed (W_v has Π), skip rotation.
        if rotation_absorbed:
            cpu_quantizer._skip_rotation = True
        v_idx, v_norms = cpu_quantizer.quantize(v_prefix)
        cpu_quantizer._skip_rotation = False
        v_centroids = cpu_quantizer.codebook.centroids[v_idx.long()]
        v_rescaled = (v_centroids * v_norms).to(torch.bfloat16)

        # Build flat tensor [1, n_kv_heads, max_seq, head_dim] with quantized prefix.
        n_kv_heads = k_prefix.shape[1]
        k_full = torch.zeros(1, n_kv_heads, max_seq, head_dim, dtype=torch.bfloat16)
        v_full = torch.zeros(1, n_kv_heads, max_seq, head_dim, dtype=torch.bfloat16)
        k_full[:, :, :prompt_len, :] = k_rescaled
        v_full[:, :, :prompt_len, :] = v_rescaled

        if paged:
            # Reshape back to paged: [max_num_blocks, n_kv_heads, block_size, head_dim]
            k_full = (
                k_full.squeeze(0).reshape(n_kv_heads, max_num_blocks, blk, head_dim).permute(1, 0, 2, 3).contiguous()
            )
            v_full = (
                v_full.squeeze(0).reshape(n_kv_heads, max_num_blocks, blk, head_dim).permute(1, 0, 2, 3).contiguous()
            )

        # Write back. Hardware handles BF16 → target dtype conversion (BFP4 etc).
        ttnn.deallocate(layer.attention.layer_past[0])
        ttnn.deallocate(layer.attention.layer_past[1])
        layer.attention.layer_past[0] = ttnn.from_torch(
            k_full,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        layer.attention.layer_past[1] = ttnn.from_torch(
            v_full,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ #
    # Device + model setup                                                 #
    # ------------------------------------------------------------------ #
    import os

    num_devices = int(os.environ.get("TT_NUM_DEVICES", 1))
    if num_devices > 1:
        print(f"Setting fabric config (FABRIC_1D) for {num_devices} devices...")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    print(f"Opening mesh device ({mesh_shape})...")
    mesh_device = ttnn.open_mesh_device(mesh_shape)

    print("Creating ModelArgs...")
    model_args = ModelArgs(
        mesh_device,
        instruct=args.instruct,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
        cache_hf=True,
    )
    if args.num_layers is not None:
        model_args.n_layers = args.num_layers

    tokenizer = model_args.tokenizer

    # Tokenize
    encoded = model_args.encode_prompt(args.prompt, instruct=args.instruct)
    prompt_len = len(encoded)
    print(f"\nPrompt  : {args.prompt!r}")
    print(f"Tokens  : {prompt_len} prompt + up to {args.max_new_tokens} new")
    assert prompt_len < args.max_seq_len, f"Prompt ({prompt_len} tok) exceeds max_seq_len ({args.max_seq_len})"

    # ------------------------------------------------------------------ #
    # Weights + TT model                                                   #
    # ------------------------------------------------------------------ #
    print("\nLoading state dict...")
    state_dict = model_args.load_state_dict()

    # Absorb rotation into W_v and W_o before model creation.
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

    from pathlib import Path
    import tempfile

    from models.tt_transformers.tt.common import PagedAttentionConfig

    dtype = ttnn.bfloat8_b
    wcache = Path(tempfile.mkdtemp(prefix="tq_prefill_weights_"))

    # Paged attention for both prefill and decode on device (no teacher-forcing).
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    print("Loading TT model (paged attention, rotation-absorbed)...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=wcache,
        paged_attention_config=paged_attention_config,
    )
    del state_dict

    # kv_cache list for prefill_forward (one [K,V] pair per layer).
    kv_cache_list = [layer.attention.layer_past for layer in tt_model.layers]

    # Page table: identity mapping (block i → page i), shape [batch=1, max_num_blocks].
    page_table_cpu = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # Prefill                                                              #
    # ------------------------------------------------------------------ #
    print("\n=== Prefill ===")

    # Prepare embedded input + RoPE matrices for the full prompt.
    # forward_prefill requires seq_len % 128 == 0 — pad with zeros to next multiple of 128.
    tokens_2d = torch.tensor([encoded])  # [1, prompt_len]
    pad_to = ((prompt_len + 127) // 128) * 128
    if pad_to > prompt_len:
        tokens_2d = torch.cat([tokens_2d, torch.zeros(1, pad_to - prompt_len, dtype=torch.long)], dim=1)
    print(f"  Padding prompt {prompt_len} → {pad_to} tokens (128-alignment)")

    # get_last_token: tile-row of the last prompt token (rounded down to 32).
    # model.forward slices out the 32 rows starting here, then runs lm_head.
    get_last_token = ((prompt_len - 1) // 32) * 32

    (
        prefill_input,
        rot_mats_global,
        rot_mats_local,
        tt_page_table,
        tt_chunk_page_table,
    ) = tt_model.prepare_inputs_prefill(tokens_2d, page_table=page_table_cpu, batch_size=1, user_id=0)

    t0 = time.perf_counter()
    tt_prefill_out = tt_model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        user_id=0,
        page_table=tt_page_table,
        get_last_token=get_last_token,
        kv_cache=kv_cache_list,
        batch_size=1,
    )
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill complete in {prefill_ms:.0f} ms (includes first-run compile)")

    # Extract logits for the last prompt token → first new token.
    # tt_prefill_out shape: [1, 1, 32, vocab_size] (1×1 mesh, lm_head sharded on last dim)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)
    prefill_logits = (
        ttnn.to_torch(tt_prefill_out, mesh_composer=mesh_composer)
        .permute(2, 1, 0, 3)  # [32, 1, 1, vocab]
        .squeeze(2)[:, 0:1, : model_args.vocab_size]  # [32, 1, vocab]
    )
    ttnn.deallocate(tt_prefill_out)

    # The last prompt token sits at row (prompt_len - 1) - get_last_token within the tile.
    last_row = (prompt_len - 1) - get_last_token
    logits_last = prefill_logits[last_row : last_row + 1, :, :]  # [1, 1, vocab]
    _, next_tok = sample_host(logits_last, temperature=0, top_p=0.8)
    first_new_tok_id = int(next_tok.squeeze().item())
    print(f"  First new token: {first_new_tok_id} → {tokenizer.decode([first_new_tok_id])!r}")

    # ------------------------------------------------------------------ #
    # Migrate prefill KV to TurboQuant pre-rescaled format                #
    # ------------------------------------------------------------------ #
    cache_dtype = ttnn.bfloat4_b if args.bfp4_cache else ttnn.bfloat16
    migrate_prefill_kv_to_turbo_quant(
        tt_model,
        prompt_len=prompt_len,
        bits=args.bits,
        mesh_device=mesh_device,
        seed=args.seed,
        rotation_absorbed=True,
        cache_dtype=cache_dtype,
        cluster_shape=model_args.cluster_shape,
        paged=True,
        block_size=paged_attention_config.block_size,
    )

    # ------------------------------------------------------------------ #
    # Attach TurboQuant caches to attention layers                        #
    # ------------------------------------------------------------------ #
    n_local_kv_heads = model_args.n_kv_heads // model_args.cluster_shape[1]
    print(
        f"\nAttaching TurboQuant {args.bits}-bit cache to {len(tt_model.layers)} layers "
        f"(kv_heads={n_local_kv_heads}, head_dim={model_args.head_dim})..."
    )
    for layer in tt_model.layers:
        tq = TTNNTurboQuantCache(
            mesh_device,
            num_layers=1,
            num_kv_heads=n_local_kv_heads,
            head_dim=model_args.head_dim,
            max_seq_len=model_args.max_seq_len,
            bits=args.bits,
            seed=args.seed,
            memory_efficient=False,  # Standard paged SDPA path
        )
        tq.rotation_absorbed = True
        layer.attention.tq_cache = tq

    # ------------------------------------------------------------------ #
    # Decode loop (free generation from position prompt_len)              #
    # ------------------------------------------------------------------ #
    print("\n=== Decode ===")

    use_trace = not args.no_trace
    eot_id = tokenizer.eos_token_id
    all_new_tokens = [first_new_tok_id]
    times = []
    current_tok_id = first_new_tok_id

    # Warmup: compile decode programs at prompt_len.
    print("Warmup (compiling decode programs)...")
    t_compile = time.perf_counter()
    host_inputs_0 = tt_model.prepare_decode_inputs_host(
        torch.tensor([first_new_tok_id], dtype=torch.int64),
        torch.tensor([prompt_len], dtype=torch.int64),
        page_table=page_table_cpu,
    )
    device_inputs_w = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
    tt_out_w, _ = tt_model.ttnn_decode_forward(*device_inputs_w)
    ttnn.deallocate(tt_out_w)
    print(f"  Compile time: {time.perf_counter() - t_compile:.1f}s")

    # Capture trace.
    if use_trace:
        print("Capturing decode trace...")
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_trace, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        print("  Trace captured.")
    else:
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
        trace_id = None
        tt_out_trace = None

    for step in range(args.max_new_tokens - 1):
        pos = prompt_len + step  # absolute sequence position of current_tok_id

        host_inputs_step = tt_model.prepare_decode_inputs_host(
            torch.tensor([current_tok_id], dtype=torch.int64),
            torch.tensor([pos], dtype=torch.int64),
            page_table=page_table_cpu,
        )
        copy_host_to_device(host_inputs_step, device_tensors=trace_inputs)

        t0 = time.perf_counter()
        if use_trace:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            tt_logits = tt_out_trace
        else:
            tt_logits, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        logits = (
            ttnn.to_torch(tt_logits, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[:1, 0:1, : model_args.vocab_size]
        )
        if not use_trace:
            ttnn.deallocate(tt_logits)

        _, next_tok = sample_host(logits, temperature=0, top_p=0.8)
        current_tok_id = int(next_tok.squeeze().item())
        all_new_tokens.append(current_tok_id)

        tok_str = tokenizer.decode([current_tok_id])
        print(f"  step {step + 1:3d} (pos {pos + 1}): {current_tok_id:7d} → {tok_str!r}  ({elapsed*1000:.0f}ms)")

        if current_tok_id == eot_id:
            print("  <|eot_id|> — stopping.")
            break

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    print("\n=== Generated text ===")
    full_output = tokenizer.decode(encoded + all_new_tokens)
    print(full_output)

    if times:
        avg_ms = sum(times) / len(times) * 1000
        cache_label = "BFP4 paged" if args.bfp4_cache else "BF16"
        mode_label = f"{args.bits}-bit TurboQuant {cache_label}" + (" (traced)" if use_trace else "")
        print(f"\n=== Performance ({mode_label}) ===")
        print(f"  Prompt tokens : {prompt_len}")
        print(f"  Generated     : {len(all_new_tokens)}")
        print(f"  Avg step time : {avg_ms:.1f} ms/tok  ({1000/avg_ms:.1f} tok/s)")
        print(f"  First step    : {times[0]*1000:.0f} ms")
        if len(times) > 1:
            warm_avg = sum(times[1:]) / len(times[1:]) * 1000
            print(f"  Warm avg      : {warm_avg:.1f} ms/tok  (step 2+)")

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #
    if use_trace:
        ttnn.release_trace(mesh_device, trace_id)

    for layer in tt_model.layers:
        tq = getattr(layer.attention, "tq_cache", None)
        if tq is not None:
            tq.deallocate()

    ttnn.close_mesh_device(mesh_device)
    print("\nDone.")


if __name__ == "__main__":
    main()
