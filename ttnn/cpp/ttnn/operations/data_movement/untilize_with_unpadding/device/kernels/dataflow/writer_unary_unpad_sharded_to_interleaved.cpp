// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Writer for HEIGHT_SHARDED (tiled) -> ROW_MAJOR INTERLEAVED untilize-with-unpadding.
//
// The height-sharded input flattens to a [global_batch * H_padded, W_padded] row space that is
// split into contiguous per-core row ranges. Every H_padded-tall band is one logical matrix whose
// first H_logical rows are real data and whose remaining rows are interior tile padding. This core
// owns the absolute padded rows [start_padded_row, start_padded_row + num_rows).
//
// The compute kernel streams the untilized rows into the output CB one tile-row at a
// time. For each row this kernel derives, from its absolute position, which matrix it belongs to
// (m) and the row within that matrix (rr). Real rows (rr < H_logical) are written to interleaved
// output page m * H_logical + rr; padding rows are skipped. Column padding is dropped by writing
// only row_size_unpadded bytes while advancing the CB read pointer by the full padded block_row_size.
//
// This makes no assumption about how matrices line up with core boundaries: a matrix may be split
// across cores, a core may hold several whole matrices, or any mix — the per-row (m, rr) walk is
// correct in every case.
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_padded_row = get_arg_val<uint32_t>(1);   // absolute first padded row owned by this core
    uint32_t num_rows = get_arg_val<uint32_t>(2);           // padded rows owned by this core (multiple of tile height)
    uint32_t matrix_h_padded = get_arg_val<uint32_t>(3);    // padded height of one matrix
    uint32_t matrix_h_logical = get_arg_val<uint32_t>(4);   // logical (real) height of one matrix
    uint32_t block_row_size = get_arg_val<uint32_t>(5);     // CB row stride in bytes (padded width)
    uint32_t row_size_unpadded = get_arg_val<uint32_t>(6);  // bytes written per row (logical width)
    uint32_t ntiles_per_row = get_arg_val<uint32_t>(7);     // tiles per tile-row (CB wait/pop granularity)

    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    const auto s = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);

    // Decompose this core's first padded row into (matrix index, row within matrix); guard against a
    // zero matrix height (should never happen) so the modulo/division below are well defined.
    uint32_t m = matrix_h_padded == 0 ? 0 : start_padded_row / matrix_h_padded;
    uint32_t rr = matrix_h_padded == 0 ? 0 : start_padded_row % matrix_h_padded;

    const uint32_t num_tile_rows = num_rows / TILE_HEIGHT;
    for (uint32_t t = 0; t < num_tile_rows; t++) {
        cb_out0.wait_front(ntiles_per_row);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();
        for (uint32_t j = 0; j < TILE_HEIGHT; j++) {
            if (rr < matrix_h_logical) {
                CoreLocalMem<uint32_t> src(l1_read_addr);
                noc.async_write(
                    src,
                    s,
                    row_size_unpadded,
                    {.offset_bytes = 0},
                    {.page_id = m * matrix_h_logical + rr, .offset_bytes = 0});
            }
            l1_read_addr += block_row_size;
            rr++;
            if (rr == matrix_h_padded) {
                rr = 0;
                m++;
            }
        }
        noc.async_write_barrier();
        cb_out0.pop_front(ntiles_per_row);
    }
}
