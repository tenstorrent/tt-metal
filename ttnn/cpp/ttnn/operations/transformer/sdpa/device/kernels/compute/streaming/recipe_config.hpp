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
// Ring and exp ring kernels: per-chunk valid key rows (recipe_tail.hpp) and the ring continuation.
constexpr bool ring = get_named_compile_time_arg_val("recipe_ring") != 0;
// Streamed ring checkpoints (the writer serves both protocols); only with fused chunks.
constexpr bool ring_stream_state_requested = get_named_compile_time_arg_val("recipe_ring_stream_state") != 0;
// Joint dense recipes with unaligned key lengths: valid primary and joint key rows (0: no key tail masking).
constexpr uint32_t k_primary_rows = get_named_compile_time_arg_val("recipe_k_primary_rows");
constexpr uint32_t k_joint_rows = get_named_compile_time_arg_val("recipe_k_joint_rows");
// QK and PV matmul subblock widths: the largest of 4, 2 and 1 dividing the K chunk and the head dim.
constexpr uint32_t qk_w = get_named_compile_time_arg_val("recipe_qk_w");
constexpr uint32_t pv_w = get_named_compile_time_arg_val("recipe_pv_w");

constexpr bool k_tail = k_primary_rows > 0;
// Fused chunks run without an attn_mask (a masked build keeps every chunk on the reduce path).
constexpr bool fused_active = fused && !mask;
// Streamed ring checkpoints rely on every restored block's first chunk being a fused chunk (it waits for each
// group's O rows); without fused chunks compute keeps the synchronous protocol.
constexpr bool ring_stream_state = ring_stream_state_requested && fused_active;

// This build's TRISC (the compute API's UNPACK/MATH/PACK macros keep only their own thread's code).
constexpr bool unpack_thread = UNPACK(true ||) false;
}  // namespace recipe
