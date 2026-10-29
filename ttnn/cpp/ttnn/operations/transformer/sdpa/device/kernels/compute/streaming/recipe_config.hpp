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
// Dense key-range calls (causal, sliding window, chunked, windowed): the reader's per-Q-chunk key range and edge
// chunk masks.
constexpr bool krange = get_named_compile_time_arg_val("recipe_krange") != 0;
// Attention sink: its page CB (0: no sink), and whether the sink tensor is BF16 (else FP32).
constexpr uint32_t sink_cb = get_named_compile_time_arg_val("recipe_sink_cb");
constexpr bool sink_bf16 = get_named_compile_time_arg_val("recipe_sink_bf16") != 0;
// Ring and exp ring kernels: per-chunk valid key rows (recipe_tail.hpp) and the ring continuation.
constexpr bool ring = get_named_compile_time_arg_val("recipe_ring") != 0;
// Causal (optionally balanced) ring attention: diagonal masking and skips (recipe_ring.hpp).
constexpr bool ring_causal = get_named_compile_time_arg_val("recipe_ring_causal") != 0;
// Ring chunked prefill: Q tiles per device in a chunk group (0: not chunked), and the group's Q tiles.
constexpr uint32_t ring_chunked = get_named_compile_time_arg_val("recipe_ring_chunked");
constexpr uint32_t ring_group_tiles = get_named_compile_time_arg_val("recipe_ring_group_tiles");
// Streamed ring checkpoints (the writer serves both protocols); only with fused chunks.
constexpr bool ring_stream_state_requested = get_named_compile_time_arg_val("recipe_ring_stream_state") != 0;
// Joint dense recipes with unaligned key lengths: valid primary and joint key rows (0: no key tail masking).
constexpr uint32_t k_primary_rows = get_named_compile_time_arg_val("recipe_k_primary_rows");
constexpr uint32_t k_joint_rows = get_named_compile_time_arg_val("recipe_k_joint_rows");
// QK and PV matmul subblock widths: the largest of 4, 2 and 1 dividing the K chunk and the head dim.
constexpr uint32_t qk_w = get_named_compile_time_arg_val("recipe_qk_w");
constexpr uint32_t pv_w = get_named_compile_time_arg_val("recipe_pv_w");

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
