// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tilize one resident L1 shard in place: `Ht` tile-rows of `Wt` tiles each, from the
// row-major input CB to the tiled output CB. Both CBs are aliased onto the shard
// buffers, so no byte moves through the NoC — the only variable is whether the
// compute kernel is SYNCHRONIZED with dataflow kernels through the CB protocol.
//
// ONE compile-time constant selects that:
//
//   no_handshake == 0   "with synchronization" (baseline)
//       The standard skeleton. A reader kernel (NCRISC) publishes the input CB, a
//       writer kernel (BRISC) retires the output CB, and this kernel runs the full
//       per-tile-row credit protocol: cb_wait_front / cb_reserve_back before the
//       tilize, cb_push_back / cb_pop_front after it.
//
//   no_handshake == 1   "without synchronization"
//       No dataflow kernels and no CB protocol at all. The data is already resident:
//       every byte of the input is present before launch and every byte of the
//       output is consumed only after it, so there is nothing to wait for and
//       nothing to credit. Tile-rows are addressed by tile INDEX from the CB base
//       pointers, which never move.
//
// The tilize call itself is identical in both arms.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);  // tiles per tile-row (one tilize call)
    constexpr uint32_t Ht = get_compile_time_arg_val(3);  // tile-rows in the shard
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr bool no_handshake = get_compile_time_arg_val(5) != 0;

    compute_kernel_hw_startup(cb_in, cb_out);
    tilize_init(cb_in, Wt, cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t r = 0; r < Ht; ++r) {
            if constexpr (no_handshake) {
                // No credits, no pointer walk: row r lives at tile index r*Wt on both
                // sides, relative to CB base pointers that are never advanced.
                tilize_block(cb_in, Wt, cb_out, r * Wt, r * Wt);
            } else {
                cb_wait_front(cb_in, Wt);
                cb_reserve_back(cb_out, Wt);
                tilize_block(cb_in, Wt, cb_out);
                cb_push_back(cb_out, Wt);
                cb_pop_front(cb_in, Wt);
            }
        }
    }

    tilize_uninit(cb_in, cb_out);
}
