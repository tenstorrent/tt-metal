#!/usr/bin/env python3
"""Needle-in-haystack accuracy test for TurboQuant KV cache.

Builds a long haystack prompt with a unique "needle" sentence inserted at
some depth, then asks a question whose answer is the needle value. Measures
whether prefill+TQ-migrate+decode can retrieve the needle. Designed to
compare Track A (full TQ) vs hybrid (TQ + FP recent-W ring) at long context.

Usage:
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    HF_HOME=/localdev/proj_sw/user_dev/hf_data \
    HF_TOKEN=... \
    TT_CACHE_PATH=/localdev/mtairum/hf/ttnn_cache/meta-llama/Llama-3.1-8B-Instruct \
    PYTHONPATH=/localdev/mtairum/tt-metal \
    python turbo_quant/test_needle_in_haystack.py \
        --haystack-len 4096 --needle-depth 0.5 \
        --tq-recent-window 0   # 0=Track A, 128=hybrid
"""
import argparse
import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, copy_host_to_device, sample_host
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from turbo_quant.quantizer import TurboQuantMSE
from turbo_quant.ttnn_integration import TTNNTurboQuantCache

# --------------------------------------------------------------------- #
# Haystack content                                                       #
# --------------------------------------------------------------------- #
# Filler is innocuous geography facts. Doesn't need to be coherent — the
# test is whether the model can retrieve the needle, not generate sense.
HAYSTACK_FILLER = (
    "The Pacific Ocean is the largest body of water on Earth, covering more area than all of the planet's land combined. "
    "The Atlantic Ocean separates the Americas from Europe and Africa and is the second largest ocean by area. "
    "The Indian Ocean is bordered by the Indian subcontinent to the north, East Africa to the west, and the Sunda Islands to the east. "
    "The Arctic Ocean is the smallest and shallowest of the world's oceans and is largely covered by sea ice. "
    "The Southern Ocean encircles Antarctica and is sometimes considered an extension of the others. "
    "Mount Everest, located in the Himalayas, is the highest peak above sea level at 8,848 meters. "
    "The Mariana Trench in the western Pacific is the deepest known point in any ocean, reaching nearly 11,000 meters. "
    "The Amazon River in South America has the largest discharge of any river in the world. "
    "The Nile River, traditionally considered the longest river, flows northward through northeast Africa. "
    "The Sahara is the largest hot desert on Earth and spans much of North Africa. "
    "Antarctica is the coldest continent and contains about ninety percent of the world's ice. "
    "The Great Barrier Reef off the coast of Australia is the largest coral reef system in the world. "
    "Lake Baikal in Siberia is the deepest and oldest freshwater lake on Earth. "
    "The Congo Basin holds the world's second-largest tropical rainforest after the Amazon. "
    "The Andes form the longest continental mountain range, running along the western edge of South America. "
)

NEEDLE_TEMPLATE = "Important: the secret access code {name} uses is {value}. Remember this."
QUESTION_TEMPLATE = "\n\nQuestion: What is the secret access code that {name} uses? Answer with just the code value, nothing else.\n\nAnswer:"

# Distractors share a name+code template that's lexically very similar to the
# target — the K vectors of these phrases differ mostly by the name token and
# the last digit. If 3-bit TQ blurs them together, attention can't disambiguate
# and the model retrieves a wrong code. Hybrid's FP ring keeps the recent
# needles bit-exact, so any needle inside W of the question is unambiguous.
DISTRACTOR_NAMES = ["Tom", "Sue", "John", "Lisa", "Carl", "Anna", "Mike", "Beth", "Paul", "Jane"]


def build_haystack_prompt(
    target_tokens, target_name, target_value, depth_frac, tokenizer, distractor_count=0, distractor_value_base=None
):
    """Build a haystack with the target needle at depth_frac plus N distractors at evenly-spaced depths.

    With distractor_count=0 this is the original single-needle test. With >0,
    inserts N distractor needles whose codes differ from the target by their
    last digit (target=banana-7421, distractors=banana-7422, banana-7423, ...).
    The target sits at depth_frac; distractors are spread across other depths.

    Returns (encoded_tokens, target_needle_pos, distractor_positions).
    """
    target_needle = NEEDLE_TEMPLATE.format(name=target_name, value=target_value)
    question = QUESTION_TEMPLATE.format(name=target_name)
    intro = "You are an assistant. Read the following document carefully, then answer the question at the end.\n\n"

    target_needle_tokens = tokenizer.encode(target_needle, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    intro_tokens = tokenizer.encode(intro, add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # Build distractor needles. Spread depths excluding the target's depth.
    distractor_needles = []
    distractor_depths = []
    if distractor_count > 0:
        if distractor_value_base is None:
            distractor_value_base = target_value  # e.g., "banana-7421" → distractors banana-7422, ...
        # Parse the trailing number out of the value
        import re

        m = re.search(r"(\d+)$", target_value)
        assert m, f"Need a trailing number in target_value to vary; got {target_value!r}"
        target_num = int(m.group(1))
        prefix = target_value[: -len(m.group(1))]
        # Generate distractor values: base+1, base+2, ... offset away from target
        for i in range(distractor_count):
            d_num = target_num + i + 1
            d_name = DISTRACTOR_NAMES[i % len(DISTRACTOR_NAMES)]
            d_value = f"{prefix}{d_num}"
            d_text = NEEDLE_TEMPLATE.format(name=d_name, value=d_value)
            distractor_needles.append(tokenizer.encode(d_text, add_special_tokens=False))
        # Place distractors at evenly spaced depths AVOIDING target's depth.
        # Use depths 0.1, 0.3, 0.7, 0.9 etc, skipping the bucket containing depth_frac.
        candidate_depths = [(i + 0.5) / (distractor_count + 1) for i in range(distractor_count + 1)]
        # Closest candidate to depth_frac is the target's bucket — drop it.
        target_bucket = min(range(len(candidate_depths)), key=lambda i: abs(candidate_depths[i] - depth_frac))
        distractor_depths = [d for i, d in enumerate(candidate_depths) if i != target_bucket]

    overhead = (
        len(bos)
        + len(intro_tokens)
        + len(target_needle_tokens)
        + sum(len(d) for d in distractor_needles)
        + len(question_tokens)
    )
    filler_budget = target_tokens - overhead
    if filler_budget < 100:
        raise ValueError(f"target_tokens={target_tokens} too small (overhead={overhead})")

    # Tokenize a chunk of filler, repeat to fill the budget.
    filler_chunk_tokens = tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False)
    chunks_needed = (filler_budget // len(filler_chunk_tokens)) + 1
    filler_tokens = (filler_chunk_tokens * chunks_needed)[:filler_budget]

    # Build a list of (depth_in_filler, kind, tokens) sorted by depth, then insert.
    inserts = [(depth_frac, "target", target_needle_tokens)]
    for d, t in zip(distractor_depths, distractor_needles):
        inserts.append((d, "distractor", t))
    inserts.sort(key=lambda x: x[0])

    haystack_tokens = []
    last_filler_idx = 0
    target_pos_in_haystack = None
    distractor_positions = []
    for depth, kind, ntokens in inserts:
        insert_at = int(len(filler_tokens) * depth)
        # Snap to never go backwards
        insert_at = max(insert_at, last_filler_idx)
        haystack_tokens.extend(filler_tokens[last_filler_idx:insert_at])
        if kind == "target":
            target_pos_in_haystack = len(haystack_tokens)
        else:
            distractor_positions.append(len(haystack_tokens))
        haystack_tokens.extend(ntokens)
        last_filler_idx = insert_at
    haystack_tokens.extend(filler_tokens[last_filler_idx:])

    full_tokens = bos + intro_tokens + haystack_tokens + question_tokens
    target_pos = len(bos) + len(intro_tokens) + target_pos_in_haystack
    distractor_global_positions = [len(bos) + len(intro_tokens) + p for p in distractor_positions]

    return full_tokens, target_pos, distractor_global_positions


def migrate_prefill_kv_to_compressed_tq(
    tt_model, prompt_len, bits, mesh_device, seed, cluster_shape, block_size, recent_window=0
):
    """Memory-efficient migrate: prefill K/V → quantized indices + norms in TQ cache.

    Replaces the pre-rescaled flow that wrote `centroid × norm` BFP8 values back
    into `layer_past` (no compression on device). Instead writes BFP4 indices and
    BFP8 norms directly into `tq_cache.k_indices_dev` / `tq_cache.k_norms_dev`,
    then frees the prefill `layer_past` — leaving only the compressed TQ cache.

    Per token per layer (32 KV heads, 128 head_dim, in standard 8B model):
      Before: layer_past BFP8 = 64 KB/token; TQ cache also allocated → ~97 KB/tok
      After:  TQ cache only = ~33 KB/token (1.9× compression vs BFP8 baseline)

    Each layer is processed independently, peak DRAM growth bounded to one
    tensor's worth (~33 MB at 16K) regardless of total context length.
    """
    head_dim = tt_model.layers[0].attention.head_dim
    cpu_quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=seed, device="cpu", dtype=torch.float32)

    num_devices = cluster_shape[1] if cluster_shape else 1
    if num_devices > 1:
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=cluster_shape)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=cluster_shape)
    else:
        composer = None
        mapper = None

    print(
        f"  Migrating {len(tt_model.layers)} layers × {prompt_len} positions to compressed TQ (memory_efficient=True)..."
    )
    for li, layer in enumerate(tt_model.layers):
        tq = layer.attention.tq_cache
        # Read K and V from layer_past (paged BFP8) to CPU and dealloc immediately.
        k_bf16 = ttnn.typecast(layer.attention.layer_past[0], ttnn.bfloat16)
        v_bf16 = ttnn.typecast(layer.attention.layer_past[1], ttnn.bfloat16)
        k_cpu = ttnn.to_torch(k_bf16, mesh_composer=composer).float()
        v_cpu = ttnn.to_torch(v_bf16, mesh_composer=composer).float()
        ttnn.deallocate(k_bf16)
        ttnn.deallocate(v_bf16)
        ttnn.deallocate(layer.attention.layer_past[0])
        ttnn.deallocate(layer.attention.layer_past[1])
        layer.attention.layer_past = None

        # Reshape paged → flat seq: [max_blocks, n_kv_heads, block_size, head_dim]
        # → [n_kv_heads, max_seq, head_dim] then add batch dim for quantizer.
        max_blocks, n_kv_heads, blk, _ = k_cpu.shape
        max_seq = max_blocks * blk
        k_flat = k_cpu.permute(1, 0, 2, 3).reshape(n_kv_heads, max_seq, head_dim).unsqueeze(0)  # [1, H, S, D]
        v_flat = v_cpu.permute(1, 0, 2, 3).reshape(n_kv_heads, max_seq, head_dim).unsqueeze(0)
        del k_cpu, v_cpu

        k_prefix = k_flat[:, :, :prompt_len, :]  # [1, H, P, D]
        v_prefix = v_flat[:, :, :prompt_len, :]

        # Quantize on CPU. K: needs rotation; V: rotation absorbed into W_v already.
        k_idx, k_norms = cpu_quantizer.quantize(k_prefix)  # idx [1,H,P,D] long, norms [1,H,P,1] float
        cpu_quantizer._skip_rotation = True
        v_idx, v_norms = cpu_quantizer.quantize(v_prefix)
        cpu_quantizer._skip_rotation = False

        # Build full-size paged tensors with zeros for [prompt_len..max_seq] —
        # the TQ cache is allocated for the full max_seq_len.
        k_idx_full = torch.zeros((1, n_kv_heads, max_seq, head_dim), dtype=torch.bfloat16)
        v_idx_full = torch.zeros_like(k_idx_full)
        k_norms_full = torch.zeros((1, n_kv_heads, max_seq, 1), dtype=torch.bfloat16)
        v_norms_full = torch.zeros_like(k_norms_full)
        # Indices stored as bf16-encoded integers (the codebook bucket id 0..2^bits-1).
        k_idx_full[:, :, :prompt_len, :] = k_idx.to(torch.bfloat16)
        v_idx_full[:, :, :prompt_len, :] = v_idx.to(torch.bfloat16)
        k_norms_full[:, :, :prompt_len, :] = k_norms.to(torch.bfloat16)
        v_norms_full[:, :, :prompt_len, :] = v_norms.to(torch.bfloat16)

        # Reshape back to paged layout: [max_blocks, n_kv_heads, block_size, head_dim].
        k_idx_paged = (
            k_idx_full.squeeze(0).reshape(n_kv_heads, max_blocks, blk, head_dim).permute(1, 0, 2, 3).contiguous()
        )
        v_idx_paged = (
            v_idx_full.squeeze(0).reshape(n_kv_heads, max_blocks, blk, head_dim).permute(1, 0, 2, 3).contiguous()
        )
        k_norms_paged = k_norms_full.squeeze(0).reshape(n_kv_heads, max_blocks, blk, 1).permute(1, 0, 2, 3).contiguous()
        v_norms_paged = v_norms_full.squeeze(0).reshape(n_kv_heads, max_blocks, blk, 1).permute(1, 0, 2, 3).contiguous()

        # Replace the zero-init device tensors with the populated ones.
        ttnn.deallocate(tq.k_indices_dev[li] if li < len(tq.k_indices_dev) else tq.k_indices_dev[0])
        ttnn.deallocate(tq.v_indices_dev[li] if li < len(tq.v_indices_dev) else tq.v_indices_dev[0])
        ttnn.deallocate(tq.k_norms_dev[li] if li < len(tq.k_norms_dev) else tq.k_norms_dev[0])
        ttnn.deallocate(tq.v_norms_dev[li] if li < len(tq.v_norms_dev) else tq.v_norms_dev[0])

        idx_dtype = ttnn.bfloat4_b
        norms_dtype = ttnn.bfloat8_b
        tq.k_indices_dev[li] = ttnn.from_torch(
            k_idx_paged,
            dtype=idx_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tq.v_indices_dev[li] = ttnn.from_torch(
            v_idx_paged,
            dtype=idx_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tq.k_norms_dev[li] = ttnn.from_torch(
            k_norms_paged,
            dtype=norms_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tq.v_norms_dev[li] = ttnn.from_torch(
            v_norms_paged,
            dtype=norms_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        del k_idx_full, v_idx_full, k_norms_full, v_norms_full
        del k_idx_paged, v_idx_paged, k_norms_paged, v_norms_paged

        # Hybrid: populate FP ring with the last W prompt positions for this layer.
        # K must be rotated by Π_K to match the TQ cache convention (which stores
        # Π·K so that Q_rot · K_stored^T = Q · K^T after rotation cancels). V is
        # already rotated because W_v has Π_V absorbed (rotation_absorbed=True).
        if recent_window > 0:
            W = recent_window
            ring_W = tq.ring_W_padded
            ring_blocks = tq.ring_blocks
            start = prompt_len - W
            k_recent = k_flat.squeeze(0)[:, start:prompt_len, :]  # [H, W, D] un-rotated
            v_recent = v_flat.squeeze(0)[:, start:prompt_len, :]  # [H, W, D] already rotated
            # Rotate K: [H, W, D] @ [D, D] = [H, W, D]
            k_recent = k_recent.float() @ cpu_quantizer.rotation
            k_ring = torch.zeros((ring_blocks, n_kv_heads, blk, head_dim), dtype=torch.bfloat16)
            v_ring = torch.zeros_like(k_ring)
            for i in range(W):
                ring_idx = (start + i) % ring_W
                blk_i = ring_idx // blk
                off = ring_idx % blk
                k_ring[blk_i, :, off, :] = k_recent[:, i, :].to(torch.bfloat16)
                v_ring[blk_i, :, off, :] = v_recent[:, i, :].to(torch.bfloat16)
            ttnn.deallocate(tq.k_ring_dev[li] if li < len(tq.k_ring_dev) else tq.k_ring_dev[0])
            ttnn.deallocate(tq.v_ring_dev[li] if li < len(tq.v_ring_dev) else tq.v_ring_dev[0])
            tq.k_ring_dev[li] = ttnn.from_torch(
                k_ring,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            tq.v_ring_dev[li] = ttnn.from_torch(
                v_ring,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
        del k_flat, v_flat
        if (li + 1) % 8 == 0:
            print(f"    layer {li + 1}/{len(tt_model.layers)} migrated")


def build_parser():
    p = argparse.ArgumentParser(description="TurboQuant needle-in-haystack accuracy test")
    p.add_argument("--haystack-len", type=int, default=4096, help="Target prompt token count")
    p.add_argument("--needle-depth", type=float, default=0.5, help="Where to insert the target needle (0..1)")
    p.add_argument("--needle-value", default="banana-7421", help="Target secret value (must end in digits)")
    p.add_argument("--target-name", default="Mary", help="Name in the target needle / question")
    p.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="Number of distractor needles with similar codes (last digit varied) at other depths",
    )
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--tq-recent-window", type=int, default=0, help="0=Track A, >0=hybrid (W tokens in FP ring)")
    p.add_argument("--max-seq-len", type=int, default=None, help="Overrides; default = haystack-len + 64")
    p.add_argument("--no-trace", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.max_seq_len is None:
        args.max_seq_len = args.haystack_len + 128
    # Round up to 128 for prefill alignment.
    args.max_seq_len = ((args.max_seq_len + 127) // 128) * 128

    num_devices = int(os.environ.get("TT_NUM_DEVICES", 1))
    if num_devices > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    print(f"Opening mesh device ({mesh_shape})...")
    mesh_device = ttnn.open_mesh_device(mesh_shape)

    print("Creating ModelArgs...")
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
        cache_hf=True,
    )
    if args.num_layers is not None:
        model_args.n_layers = args.num_layers

    tokenizer = model_args.tokenizer

    # Build haystack prompt.
    encoded, needle_pos, distractor_positions = build_haystack_prompt(
        args.haystack_len,
        args.target_name,
        args.needle_value,
        args.needle_depth,
        tokenizer,
        distractor_count=args.distractors,
    )
    prompt_len = len(encoded)
    needle_depth_actual = needle_pos / prompt_len
    print(f"\nPrompt        : {prompt_len} tokens  (target {args.haystack_len})")
    print(
        f"Target needle : {args.target_name}={args.needle_value!r} at token {needle_pos} (depth {needle_depth_actual:.3f})"
    )
    if args.distractors > 0:
        print(
            f"Distractors   : {args.distractors} similar needles at depths {[round(p / prompt_len, 3) for p in distractor_positions]}"
        )
    print(f"Mode          : {'HYBRID' if args.tq_recent_window > 0 else 'TRACK A'}  W={args.tq_recent_window}")
    assert prompt_len < args.max_seq_len, f"Prompt ({prompt_len}) >= max_seq_len ({args.max_seq_len})"

    print("\nLoading state dict...")
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

    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=args.max_seq_len // 32)
    wcache = Path(tempfile.mkdtemp(prefix="tq_needle_weights_"))
    print("Loading TT model (paged attention, rotation-absorbed)...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=wcache,
        paged_attention_config=paged_attention_config,
    )
    del state_dict

    kv_cache_list = [layer.attention.layer_past for layer in tt_model.layers]
    page_table_cpu = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).unsqueeze(0)

    # ----- Prefill -----
    print("\n=== Prefill ===")
    tokens_2d = torch.tensor([encoded])
    pad_to = ((prompt_len + 127) // 128) * 128
    if pad_to > prompt_len:
        tokens_2d = torch.cat([tokens_2d, torch.zeros(1, pad_to - prompt_len, dtype=torch.long)], dim=1)
    print(f"  Padded {prompt_len} → {pad_to} tokens")
    get_last_token = ((prompt_len - 1) // 32) * 32

    (prefill_input, rot_g, rot_l, tt_pt, tt_chunk_pt) = tt_model.prepare_inputs_prefill(
        tokens_2d, page_table=page_table_cpu, batch_size=1, user_id=0
    )
    t0 = time.perf_counter()
    tt_prefill_out = tt_model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_g,
        rot_mats_local=rot_l,
        user_id=0,
        page_table=tt_pt,
        get_last_token=get_last_token,
        kv_cache=kv_cache_list,
        batch_size=1,
    )
    print(f"  Prefill in {(time.perf_counter()-t0)*1000:.0f} ms")

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)
    prefill_logits = (
        ttnn.to_torch(tt_prefill_out, mesh_composer=composer)
        .permute(2, 1, 0, 3)
        .squeeze(2)[:, 0:1, : model_args.vocab_size]
    )
    ttnn.deallocate(tt_prefill_out)
    for t in [prefill_input, tt_pt]:
        if t is not None:
            ttnn.deallocate(t)
    for rot in (rot_g, rot_l):
        if rot is not None:
            for t in rot if isinstance(rot, (list, tuple)) else [rot]:
                if t is not None:
                    ttnn.deallocate(t)

    last_row = (prompt_len - 1) - get_last_token
    _, next_tok = sample_host(prefill_logits[last_row : last_row + 1, :, :], temperature=0, top_p=0.8)
    first_new_tok_id = int(next_tok.squeeze().item())
    print(f"  First new token: {first_new_tok_id} → {tokenizer.decode([first_new_tok_id])!r}")

    # ----- Allocate compressed TQ cache (memory_efficient=True) -----
    n_local_kv_heads = model_args.n_kv_heads // model_args.cluster_shape[1]
    print(
        f"\nAllocating {args.bits}-bit compressed TQ cache (memory_efficient=True, "
        f"recent_window={args.tq_recent_window})..."
    )
    shared_tq = TTNNTurboQuantCache(
        mesh_device,
        num_layers=len(tt_model.layers),
        num_kv_heads=n_local_kv_heads,
        head_dim=model_args.head_dim,
        max_seq_len=args.max_seq_len,
        bits=args.bits,
        memory_efficient=True,  # paged BFP4 indices + BFP8 norms + fused SDPA
        paged_config=paged_attention_config,
        max_batch_size=1,
        seed=args.seed,
        recent_window=args.tq_recent_window,
    )
    shared_tq.rotation_absorbed = True
    for layer_idx, layer in enumerate(tt_model.layers):
        layer.attention.tq_cache = shared_tq
        layer.attention.tq_layer_idx = layer_idx
    if args.tq_recent_window > 0:
        tt_model.tq_ring_W = shared_tq.recent_window
        tt_model.tq_ring_W_padded = shared_tq.ring_W_padded

    # ----- Migrate prefill KV → compressed TQ (writes indices+norms, frees layer_past) -----
    # If hybrid (recent_window > 0), also populates the FP ring from the last
    # W prompt positions in the same pass — done inline because layer_past is
    # freed per layer and can't be re-read afterwards.
    migrate_prefill_kv_to_compressed_tq(
        tt_model,
        prompt_len=prompt_len,
        bits=args.bits,
        mesh_device=mesh_device,
        seed=args.seed,
        cluster_shape=model_args.cluster_shape,
        block_size=paged_attention_config.block_size,
        recent_window=args.tq_recent_window,
    )

    # ----- Decode -----
    print("\n=== Decode ===")
    use_trace = not args.no_trace
    eot_id = tokenizer.eos_token_id
    all_new_tokens = [first_new_tok_id]
    times = []
    current_tok_id = first_new_tok_id

    print("Warmup (compiling decode programs)...")
    host_inputs_0 = tt_model.prepare_decode_inputs_host(
        torch.tensor([first_new_tok_id], dtype=torch.int64),
        torch.tensor([prompt_len], dtype=torch.int64),
        page_table=page_table_cpu,
    )
    device_inputs_w = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
    tt_out_w, _ = tt_model.ttnn_decode_forward(*device_inputs_w)
    ttnn.deallocate(tt_out_w)

    if use_trace:
        print("Capturing trace...")
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_trace, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    else:
        trace_inputs = copy_host_to_device(host_inputs_0, mesh_device=mesh_device)
        trace_id = None
        tt_out_trace = None

    for step in range(args.max_new_tokens - 1):
        pos = prompt_len + step
        host_step = tt_model.prepare_decode_inputs_host(
            torch.tensor([current_tok_id], dtype=torch.int64),
            torch.tensor([pos], dtype=torch.int64),
            page_table=page_table_cpu,
        )
        copy_host_to_device(host_step, device_tensors=trace_inputs)

        t0 = time.perf_counter()
        if use_trace:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            tt_logits = tt_out_trace
        else:
            tt_logits, _ = tt_model.ttnn_decode_forward(*trace_inputs)
        times.append(time.perf_counter() - t0)

        logits = (
            ttnn.to_torch(tt_logits, mesh_composer=composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[:1, 0:1, : model_args.vocab_size]
        )
        if not use_trace:
            ttnn.deallocate(tt_logits)
        _, next_tok = sample_host(logits, temperature=0, top_p=0.8)
        current_tok_id = int(next_tok.squeeze().item())
        all_new_tokens.append(current_tok_id)
        if current_tok_id == eot_id:
            break

    if use_trace:
        ttnn.release_trace(mesh_device, trace_id)

    # ----- Score -----
    generated_text = tokenizer.decode(all_new_tokens)
    print(f"\nGenerated: {generated_text!r}")
    found = args.needle_value in generated_text

    # Distractor confusion check: did the model output a distractor's code?
    distractor_hit = None
    if args.distractors > 0 and not found:
        import re

        m = re.search(r"(\d+)$", args.needle_value)
        target_num = int(m.group(1))
        prefix = args.needle_value[: -len(m.group(1))]
        for i in range(args.distractors):
            d_value = f"{prefix}{target_num + i + 1}"
            if d_value in generated_text:
                distractor_hit = d_value
                break

    if found:
        result = "PASS_EXACT"
    elif distractor_hit:
        result = f"FAIL_DISTRACTOR(retrieved {distractor_hit})"
    else:
        result = "FAIL"

    print(f"\n{'=' * 60}")
    print(f"TARGET           : {args.target_name}={args.needle_value!r}")
    print(f"GENERATED TEXT   : {generated_text!r}")
    print(f"NEEDLE DEPTH     : {needle_depth_actual:.3f} (token {needle_pos}/{prompt_len})")
    print(f"MODE             : {'HYBRID W=' + str(args.tq_recent_window) if args.tq_recent_window > 0 else 'TRACK A'}")
    print(f"DISTRACTORS      : {args.distractors}")
    print(f"RESULT           : {result}")
    print(f"{'=' * 60}")

    for layer in tt_model.layers:
        tq = getattr(layer.attention, "tq_cache", None)
        if tq is not None:
            tq.deallocate()
    ttnn.close_mesh_device(mesh_device)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
