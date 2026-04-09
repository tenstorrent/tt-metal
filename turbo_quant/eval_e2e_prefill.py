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
    p.add_argument("--no-trace", action="store_true", help="Disable TTNN trace (slower, useful for debugging)")
    return p


def migrate_prefill_kv_to_turbo_quant(
    tt_model,
    tq_caches,
    prompt_len: int,
    bits: int,
    mesh_device,
    seed: int = 42,
):
    """Copy prefill KV from layer_past into TurboQuant CPU shadow buffers.

    After a prefill forward pass, each attention layer's layer_past holds the
    full-context key/value vectors in BFP8 format.  This function:
      1. Typecasts each layer_past slab BFP8 → BF16 on device.
      2. Brings the valid prefix [0..prompt_len-1] to CPU as float32.
      3. Quantizes with TurboQuantMSE (same seed as the TTNN setup).
      4. Stores the resulting indices + norms in the tq_cache device buffers so
         that the first decode step at position prompt_len sees the full
         prefill context.

    Args:
        tt_model: Loaded Transformer with layer_past filled by prefill.
        tq_caches: List of TTNNTurboQuantCache, one per layer.
        prompt_len: Number of prompt tokens (valid positions in layer_past).
        bits: Quantisation bit-width (must match tq_cache settings).
        mesh_device: TTNN mesh device (needed to upload BF16 index tensors).
        seed: Rotation matrix seed (must match tq_cache settings).
    """
    head_dim = tt_model.layers[0].attention.head_dim
    cpu_quantizer = TurboQuantMSE(
        head_dim=head_dim,
        bits=bits,
        seed=seed,
        device="cpu",
        dtype=torch.float32,
    )

    print(f"  Migrating {len(tt_model.layers)} layers × {prompt_len} positions to TurboQuant ...")
    for layer, tq_cache in zip(tt_model.layers, tq_caches):
        # Cast BFP8 KV cache to BF16 on device, then pull to CPU.
        # layer_past shape: [batch, n_local_kv_heads, max_seq_len, head_dim]
        k_bf16 = ttnn.typecast(layer.attention.layer_past[0], ttnn.bfloat16)
        v_bf16 = ttnn.typecast(layer.attention.layer_past[1], ttnn.bfloat16)

        # to_torch on a 1×1 mesh tensor works without a composer.
        k_cpu = ttnn.to_torch(k_bf16).float()  # [1, heads, max_seq, head_dim]
        v_cpu = ttnn.to_torch(v_bf16).float()
        ttnn.deallocate(k_bf16)
        ttnn.deallocate(v_bf16)

        # Take only the filled prefix.
        k_prefix = k_cpu[:, :, :prompt_len, :]  # [1, heads, prompt_len, head_dim]
        v_prefix = v_cpu[:, :, :prompt_len, :]

        # CPU quantize (same rotation + codebook as on-device path).
        k_idx, k_norms = cpu_quantizer.quantize(k_prefix)  # uint8 [..., head_dim], float [..., 1]
        v_idx, v_norms = cpu_quantizer.quantize(v_prefix)

        # Convert indices to the format expected by the cache.
        # If cache_centroids=True: store centroid values (via CPU dequantize).
        # Otherwise: store integer indices as BF16.
        max_seq_padded = tq_cache.k_indices_dev[0].shape[2]
        if getattr(tq_cache, "cache_centroids", False):
            # Store pre-rescaled centroid × norm values (matches decode-time cache format).
            k_centroids = cpu_quantizer.codebook.centroids[k_idx.long()]  # [..., head_dim]
            v_centroids = cpu_quantizer.codebook.centroids[v_idx.long()]
            k_idx_bf16 = (k_centroids * k_norms).to(torch.bfloat16)
            v_idx_bf16 = (v_centroids * v_norms).to(torch.bfloat16)
        else:
            k_idx_bf16 = k_idx.float().to(torch.bfloat16)
            v_idx_bf16 = v_idx.float().to(torch.bfloat16)

        pad_len = max_seq_padded - prompt_len
        if pad_len > 0:
            pad = torch.zeros(1, k_idx_bf16.shape[1], pad_len, k_idx_bf16.shape[3], dtype=torch.bfloat16)
            k_idx_padded = torch.cat([k_idx_bf16, pad], dim=2)
            v_idx_padded = torch.cat([v_idx_bf16, pad], dim=2)
        else:
            k_idx_padded = k_idx_bf16
            v_idx_padded = v_idx_bf16

        ttnn.deallocate(tq_cache.k_indices_dev[0])
        ttnn.deallocate(tq_cache.v_indices_dev[0])
        tq_cache.k_indices_dev[0] = ttnn.from_torch(
            k_idx_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tq_cache.v_indices_dev[0] = ttnn.from_torch(
            v_idx_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Phase 1.5-A2: norms live on device as BF16.
        # Pad to max_seq_padded (same as index tensor), deallocate old, upload new.
        max_seq_padded = tq_cache.k_norms_dev[0].shape[2]
        k_norms_padded = torch.zeros(1, k_norms.shape[1], max_seq_padded, 1, dtype=torch.bfloat16)
        v_norms_padded = torch.zeros(1, v_norms.shape[1], max_seq_padded, 1, dtype=torch.bfloat16)
        k_norms_padded[:, :, :prompt_len, :] = k_norms.float().to(torch.bfloat16)
        v_norms_padded[:, :, :prompt_len, :] = v_norms.float().to(torch.bfloat16)

        ttnn.deallocate(tq_cache.k_norms_dev[0])
        ttnn.deallocate(tq_cache.v_norms_dev[0])
        tq_cache.k_norms_dev[0] = ttnn.from_torch(
            k_norms_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tq_cache.v_norms_dev[0] = ttnn.from_torch(
            v_norms_padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ #
    # Device + model setup                                                 #
    # ------------------------------------------------------------------ #
    print("Opening mesh device (1×1)...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

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

    dtype = ttnn.bfloat8_b
    print("Loading TT model (non-paged, required for TurboQuant decode path)...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=None,
    )
    del state_dict

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
    ) = tt_model.prepare_inputs_prefill(tokens_2d)

    t0 = time.perf_counter()
    tt_prefill_out = tt_model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        page_table=tt_page_table,
        get_last_token=get_last_token,
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
    # Build TurboQuant caches (one per layer, each wrapping 1 layer)      #
    # ------------------------------------------------------------------ #
    n_local_kv_heads = model_args.n_kv_heads // model_args.cluster_shape[1]
    print(
        f"\nBuilding TurboQuant {args.bits}-bit caches "
        f"({len(tt_model.layers)} layers, "
        f"kv_heads={n_local_kv_heads}, head_dim={model_args.head_dim})..."
    )
    tq_caches = [
        TTNNTurboQuantCache(
            mesh_device,
            num_layers=1,
            num_kv_heads=n_local_kv_heads,
            head_dim=model_args.head_dim,
            max_seq_len=model_args.max_seq_len,
            bits=args.bits,
            seed=args.seed,
        )
        for _ in tt_model.layers
    ]

    # ------------------------------------------------------------------ #
    # Migrate prefill KV into TurboQuant shadow buffers                   #
    # ------------------------------------------------------------------ #
    migrate_prefill_kv_to_turbo_quant(
        tt_model, tq_caches, prompt_len=prompt_len, bits=args.bits, mesh_device=mesh_device, seed=args.seed
    )

    # Attach caches to attention layers for the decode path.
    for layer, tq_cache in zip(tt_model.layers, tq_caches):
        layer.attention.tq_cache = tq_cache

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
        mode_label = f"{args.bits}-bit TurboQuant" + (" (traced)" if use_trace else "")
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

    for tq_cache in tq_caches:
        tq_cache.deallocate()

    ttnn.close_mesh_device(mesh_device)
    print("\nDone.")


if __name__ == "__main__":
    main()
