// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/matmul.h"

/**
 * @file matmul_helpers.hpp
 * @brief Row-run matmul helpers for compute kernels
 *
 * matmul_tiles unpacks both operands for every product. When a run of output tiles in one output row
 * shares its left operand tile, matmul_tiles_row unpacks that tile once and streams the run's right
 * operand tiles past it. for_each_row_run splits a DST batch of output tiles into such runs.
 */

namespace compute_kernel_lib {

/**
 * @brief DST[idst + c] += in0[in0_tile_index] @ in1[in1_tile_index + c] for c < n
 *
 * Same result as n matmul_tiles calls on n consecutive in1 tiles of one in0 tile, with in0 unpacked ONCE
 * (it stays in SrcB while the n in1 tiles stream through SrcA) instead of once per product. Each DST tile
 * receives exactly the MVMUL sequence matmul_tiles issues for it, so the sums are bit-identical.
 *
 * Preconditions, as for matmul_tiles:
 * - matmul_init (or matmul_block_init) has configured the unpacker and math engine for in0 / in1.
 * - DST is acquired and DST tiles [idst, idst + n) fit in it.
 * - in1's tile size is the unpacker's configured SrcA tile size (the unpacker steps from one in1 tile to
 *   the next by it).
 *
 * This is matmul_tiles's own unpack / math pair given a column count, not matmul_block: on Blackhole
 * matmul_block switches to the firmware-throttled MOP on request, whose reordered fidelity phases would
 * make the sums depend on the throttle state. On Quasar it falls back to n matmul_tiles calls.
 *
 * @param in0_cb_id       CB holding the shared left operand tile
 * @param in1_cb_id       CB holding the right operand tiles
 * @param in0_tile_index  Index of the left operand tile in in0_cb_id
 * @param in1_tile_index  Index of the first of the n consecutive right operand tiles in in1_cb_id
 * @param idst            First DST tile of the run
 * @param n               Number of output tiles in the run
 */
ALWI void matmul_tiles_row(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    uint32_t n) {
#ifndef ARCH_QUASAR
    LLK_SAN_FUNCTION();
    UNPACK((llk_unpack_AB_matmul(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, n /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, n /*ct_dim*/, 1 /*rt_dim*/)));
#else
    for (uint32_t c = 0; c < n; c++) {
        ckernel::matmul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index + c, idst + c);
    }
#endif
}

/**
 * @brief Split a DST batch of output tiles into runs that share an output row
 *
 * The batch is output tiles [t0, t0 + nb) of an [M, Nt] tile grid in row-major order, held in DST tiles
 * [0, nb). Calls f(mi, ni, j, n) for each run: the run starts at grid tile (mi, ni), DST tile j, and is n
 * tiles long. Pair it with matmul_tiles_row to unpack each left operand tile once per run.
 */
template <typename F>
ALWI void for_each_row_run(uint32_t t0, uint32_t nb, uint32_t Nt, F&& f) {
    for (uint32_t j = 0; j < nb;) {
        const uint32_t t = t0 + j;
        const uint32_t mi = t / Nt;
        const uint32_t ni = t - mi * Nt;
        const uint32_t n = (nb - j < Nt - ni) ? (nb - j) : (Nt - ni);
        f(mi, ni, j, n);
        j += n;
    }
}

}  // namespace compute_kernel_lib
