// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

namespace matmul_block_config {

// Default no-op post-compute functor.
// Called per output sub-block on the last K-block, before packing.
// Receives out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

// Hardware RELU activation applied by the packer during pack (zero cost).
// Pass as PostComputeFn to use hardware RELU instead of SFPU-based activation.
// The helper detects this type and configures the packer RELU register
// instead of calling SFPU instructions.
struct HwRelu {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

// Default no-op pre-K-block functor.
// Called at the start of each K-block iteration, before input CB waits.
// Receives (block_index, num_k_blocks, is_last_block).
// Use for per-K-block preprocessing (e.g., in0_transpose, global CB pointer manipulation).
struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

}  // namespace matmul_block_config

/**
 * matmul_block: sub-blocked tiled matrix multiplication C = A × B with K-blocking.
 *
 * Performs matrix multiplication using matmul_block LLK with sub-block indexing
 * and automatic K-dimension blocking. Supports two blocking strategies selected
 * at compile time:
 *
 *   packer_l1_acc=false: Software spill/reload via interm_cb (default)
 *   packer_l1_acc=true:  Hardware L1 accumulation via packer (avoids spill/reload)
 *
 * PREREQUISITE: Caller must call mm_block_init() before invoking this helper.
 * The helper does NOT call mm_block_init internally.
 *
 * Uses 4-phase DST management (tile_regs_acquire/commit/wait/release) for
 * correct MATH-PACK pipelining, matching the production kernel and all other
 * kernel_lib helpers.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   in0_cb            Input CB for matrix A (0–31).
 *   in1_cb            Input CB for matrix B (0–31).
 *   out_cb            Output CB for result C (0–31). Also used for shared memory
 *                     protection with interm_cb (they overlap in L1).
 *   interm_cb         Intermediate CB for K-blocking (0–31). Used for spill/reload
 *                     (software) or L1 accumulation (hardware). Must differ from out_cb.
 *   transpose         If true, transpose B tiles before multiplication (default: false).
 *   packer_l1_acc     If true, use packer L1 accumulation instead of software
 *                     spill/reload. (default: false)
 *   pack_last_to_interm  If true, the last K-block packs to interm_cb instead of
 *                     out_cb. Use when a post-processing phase (bias add, untilize)
 *                     reads from interm_cb. (default: false)
 *   PostComputeFn     Activation applied per output sub-block on the last K-block.
 *                     Pass HwRelu for zero-cost packer hardware RELU.
 *                     Pass a custom functor for SFPU activations (gelu, silu, etc.).
 *                     HwRelu cannot be used with pack_last_to_interm (enforced
 *                     by static_assert). (default: NoPostCompute)
 *   PreKBlockFn       Functor called at the start of each K-block iteration,
 *                     before input CB waits. Receives (block, num_k_blocks,
 *                     is_last). Use for per-K-block preprocessing such as
 *                     in0_transpose. (default: NoPreKBlock)
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   block_w            Inner block dimension in tiles (K-dimension block size).
 *   in0_num_subblocks  Number of sub-blocks along the M dimension.
 *   in1_num_subblocks  Number of sub-blocks along the N dimension.
 *   num_k_blocks       Number of blocks along the K dimension.
 *   out_subblock_h     Output sub-block height in tiles.
 *   out_subblock_w     Output sub-block width in tiles.
 *   batch              Number of independent batch slices (default: 1).
 *   post_compute       PostComputeFn instance (default: {}).
 *   pre_k_block        PreKBlockFn instance (default: {}).
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    typename PostComputeFn = matmul_block_config::NoPostCompute,
    typename PreKBlockFn = matmul_block_config::NoPreKBlock>
ALWI void matmul_block(
    const uint32_t block_w,
    const uint32_t in0_num_subblocks,
    const uint32_t in1_num_subblocks,
    const uint32_t num_k_blocks,
    const uint32_t out_subblock_h,
    const uint32_t out_subblock_w,
    const uint32_t batch = 1,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {});

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
