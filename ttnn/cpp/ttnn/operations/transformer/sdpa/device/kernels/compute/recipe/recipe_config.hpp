// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// The recipe compute kernels' build configuration: named compile-time arguments the host sets on every recipe compute
// kernel (append_recipe_compute_args, sdpa_recipe.hpp). Kernels select behavior with if constexpr on these.
namespace recipe {
// BALANCED and ACCURATE: FP32 DEST and FP32 online-softmax state.
constexpr bool fp32 = get_named_compile_time_arg_val("recipe_fp32") != 0;
// ACCURATE (one of the FP32 recipes).
constexpr bool accurate = get_named_compile_time_arg_val("recipe_accurate") != 0;
// FAST: LoFi matmuls.
constexpr bool lofi = get_named_compile_time_arg_val("recipe_lofi") != 0;
// Fused K chunks (STANDARD and FAST, when their CBs fit L1).
constexpr bool fused = get_named_compile_time_arg_val("recipe_fused") != 0;
// Dense additive attn_mask.
constexpr bool mask = get_named_compile_time_arg_val("recipe_mask") != 0;
// The attn_mask's factor 1 / scale (FP32 bits; 0: none): the exps scale masked scores by `scale`, so each mask tile is
// multiplied by its inverse (in FP32, on the pack thread) before the L1 add, and the mask enters softmax unscaled.
constexpr uint32_t mask_scale = get_named_compile_time_arg_val("recipe_mask_scale");
// Dense key-range calls (causal, sliding window, chunked, windowed): the reader's per-Q-chunk key range and edge
// chunk masks.
constexpr bool krange = get_named_compile_time_arg_val("recipe_krange") != 0;
// Attention sink: its page CB (0: no sink), and whether the sink tensor is BF16 (else FP32).
constexpr uint32_t sink_cb = get_named_compile_time_arg_val("recipe_sink_cb");
constexpr bool sink_bf16 = get_named_compile_time_arg_val("recipe_sink_bf16") != 0;
// Ring and exp ring kernels: per-chunk key masks (recipe_tail.hpp: RecipeRingMask) and the ring continuation.
constexpr bool ring = get_named_compile_time_arg_val("recipe_ring") != 0;
// Ring positional masks (recipe_tail.hpp): 0 none (only key padding), 1 causal in the device's shard frame (Q and K
// tiles both counted from the shard's start), 2 the sequence frame (chunked prefill, sliding windows, KV-pad rotation:
// absolute positions, a local K tile mapped through the chunk groups' regions).
constexpr uint32_t ring_mask = get_named_compile_time_arg_val("recipe_ring_mask");
// Causal sliding window in keys (0: none): key k is visible to query q iff q - window < k <= q.
constexpr uint32_t ring_window = get_named_compile_time_arg_val("recipe_ring_window");
// KV-pad rotation: Q rows map to the sequence through the slab's wrap segments; K tiles past the cache or the logical
// length are masked.
constexpr bool ring_rotation = get_named_compile_time_arg_val("recipe_ring_rotation") != 0;
// Circular sliding KV cache: a K chunk's sequence position is given whole (several chunk groups alias a slab).
constexpr bool ring_circular = get_named_compile_time_arg_val("recipe_ring_circular") != 0;
// Streamed ring checkpoints (the writer serves both protocols); only with fused chunks.
constexpr bool ring_stream_state_requested = get_named_compile_time_arg_val("recipe_ring_stream_state") != 0;
// Joint dense recipes with unaligned key lengths: valid primary and joint key rows (0: no key tail masking).
constexpr uint32_t k_primary_rows = get_named_compile_time_arg_val("recipe_k_primary_rows");
constexpr uint32_t k_joint_rows = get_named_compile_time_arg_val("recipe_k_joint_rows");
// QK and PV matmul subblock widths: the largest of 4, 2 and 1 dividing the K chunk and the head dim.
constexpr uint32_t qk_w = get_named_compile_time_arg_val("recipe_qk_w");
constexpr uint32_t pv_w = get_named_compile_time_arg_val("recipe_pv_w");
// STANDARD, FAST: the most K tiles a PV accumulates in DEST before the packer adds it to O in FP32 (the policy's
// pv_fold_tiles; 0: no limit).
constexpr uint32_t pv_fold = get_named_compile_time_arg_val("recipe_pv_fold");

constexpr bool k_tail = k_primary_rows > 0;
constexpr bool sink = sink_cb != 0;
// Fused chunks run without an attn_mask (a masked build keeps every chunk on the reduce path); key-range calls mask
// only their edge K chunks, so their full chunks stay fused.
constexpr bool fused_active = fused && (!mask || krange);
// Streamed ring checkpoints rely on every restored block's first chunk being a fused chunk (it waits for each
// group's O rows); without fused chunks compute keeps the synchronous protocol.
constexpr bool ring_stream_state = ring_stream_state_requested && fused_active;

// This build's TRISC (the compute API's UNPACK/MATH/PACK macros keep only their own thread's code).
constexpr bool unpack_thread = UNPACK(true ||) false;
constexpr bool pack_thread = PACK(true ||) false;

// The architecture: the JIT build's ARCH_* define, read here once so the recipe code selects with if constexpr.
#ifdef ARCH_WORMHOLE
constexpr bool wormhole = true;
#else
constexpr bool wormhole = false;
#endif
}  // namespace recipe
