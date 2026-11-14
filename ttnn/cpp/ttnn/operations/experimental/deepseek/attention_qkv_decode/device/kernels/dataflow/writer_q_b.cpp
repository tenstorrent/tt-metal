// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"
#include "rope_table_rows.hpp"

// A q_b core: the RoPE table rows its compute needs (do_rope: it holds a head's last tiles), then its
// Nc result tiles -- row-major bf16, a 1x32 tile is 32 contiguous values -- into row q_row of q at
// byte q_col_bytes.
//
// COMP (on is_comp cores): before q, its compressor-column tile into comp_output's one row at byte
// comp_col_bytes.
void kernel_main() {
    constexpr uint32_t nc_tiles = get_arg(args::nc_tiles);
    constexpr uint32_t rope_tiles = get_arg(args::rope_tiles);
    constexpr uint32_t row_tile_bytes = get_arg(args::row_tile_bytes);
    const bool do_rope = get_arg(args::do_rope) != 0;
    const uint32_t q_row = get_arg(args::q_row);
    const uint32_t q_col_bytes = get_arg(args::q_col_bytes);

    Noc noc;
    CircularBuffer cs_cb(dfb::cs);
    CircularBuffer cs_raw_cb(dfb::cs_raw);
    CircularBuffer out_cb(dfb::out);

    if (do_rope) {
        const auto cos = TensorAccessor(tensor::cos);
        const auto sin = TensorAccessor(tensor::sin);
        read_rope_table_rows(noc, cos, sin, 0, rope_tiles, cs_cb, cs_raw_cb, row_tile_bytes);
    }

#ifdef COMP
    if (get_arg(args::is_comp) != 0) {
        CircularBuffer comp_x_cb(dfb::comp_x);
        const auto comp_output = TensorAccessor(tensor::comp_output);
        comp_x_cb.wait_front(1);
        noc_async_write(
            comp_x_cb.get_read_ptr(), comp_output.get_noc_addr(0, get_arg(args::comp_col_bytes)), row_tile_bytes);
        noc.async_write_barrier();
        comp_x_cb.pop_front(1);
    }
#endif

    const auto q = TensorAccessor(tensor::q);
    out_cb.wait_front(nc_tiles);
    noc_async_write(out_cb.get_read_ptr(), q.get_noc_addr(q_row, q_col_bytes), nc_tiles * row_tile_bytes);
    noc.async_write_barrier();
    out_cb.pop_front(nc_tiles);
}
