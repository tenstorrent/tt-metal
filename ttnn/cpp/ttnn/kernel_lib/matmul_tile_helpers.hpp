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

namespace matmul_tile_config {

// Default no-op post-compute functor.
// Called per output tile after the Kt accumulation loop, before packing.
// Receives 1 (the single tile in DST[0]).
struct NoPostCompute {
    ALWI void operator()(uint32_t /* num_tiles */) const {}
};

}  // namespace matmul_tile_config

/**
 * matmul_tile: tile-by-tile matrix multiplication C = A x B using matmul_tiles LLK.
 *
 * Performs tile-by-tile matrix multiplication. For each output tile C[mt,nt],
 * accumulates Kt inner products: C[mt,nt] = sum_kt A[mt,kt] * B[kt,nt].
 *
 * Loop order: batch x Mt x Nt x Kt (must match CB production order from reader).
 *
 * CB synchronization: cb_wait_front/cb_pop_front per tile for both in0 and in1.
 * The reader must push tiles one at a time in the matching loop order.
 *
 * Uses 4-phase DST management (tile_regs_acquire/commit/wait/release) for
 * correct MATH-PACK pipelining, matching matmul_block and all other helpers.
 *
 * PREREQUISITE: Caller must call mm_init() before invoking this helper.
 * The helper does NOT call mm_init internally.
 *
 * -- Template Parameters --
 *
 *   in0_cb          Input CB for matrix A (0-31).
 *   in1_cb          Input CB for matrix B (0-31).
 *   out_cb          Output CB for result C (0-31).
 *   transpose       If true, transpose B tiles before multiplication (default: false).
 *   PostComputeFn   Functor called per output tile after the Kt accumulation loop,
 *                   before packing. Receives num_tiles (always 1). (default: NoPostCompute)
 *
 * -- Runtime Parameters --
 *
 *   Mt              Number of output tile rows.
 *   Nt              Number of output tile columns.
 *   Kt              Number of inner-dimension tiles.
 *   batch           Number of independent batch slices (default: 1).
 *   post_compute    PostComputeFn instance (default: {}).
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    bool transpose = false,
    typename PostComputeFn = matmul_tile_config::NoPostCompute>
ALWI void matmul_tile(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch = 1, PostComputeFn post_compute = {});

}  // namespace compute_kernel_lib

#include "matmul_tile_helpers.inl"
