// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// padded_slice gather (op_design.md, regime A): this core's result rows, read from the input's DRAM pages
// straight into their place in this core's own output shard.
//
// No kernel-library helper applies: the op has no compute phase (spec: "moves values and computes
// nothing"), and the one dataflow helper (mcast) moves one payload to many cores, which nothing here does.
// The data movement is the Noc / TensorAccessor / LocalTensorAccessor API directly.
//
// Rows are the result's rows [first_row, first_row + num_rows) (per-node RT args). They are taken in blocks
// of `block_rows` (the block knob; the whole shard by default); inside a block hart t of R takes the
// contiguous sub-range [lo_t, hi_t). Per hart per block: one (n, h, w) decomposition, then counters; one
// tagged NoC read per row (an interleaved page is the transfer quantum: consecutive pages live in different
// banks); one barrier.
//
// No ring: every read carries the user transaction id (trid 0) and the barrier waits on it, so the hart
// neither inherits a ring's tag from an implicit-sync program before it nor leaves one for an untagged
// program after it. The tag register is handed back as the last statement.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_row = get_arg(args::first_row);
    const uint32_t num_rows = get_arg(args::num_rows);

    const uint32_t block_rows = get_arg(args::block_rows);
    const uint32_t block_h = get_arg(args::block_h);
    const uint32_t block_w = get_arg(args::block_w);
    const uint32_t in_h = get_arg(args::in_h);
    const uint32_t in_w = get_arg(args::in_w);
    const uint32_t start_n = get_arg(args::start_n);
    const uint32_t start_h = get_arg(args::start_h);
    const uint32_t start_w = get_arg(args::start_w);
    const uint32_t src_offset_bytes = get_arg(args::src_offset_bytes);
    const uint32_t row_bytes = get_arg(args::row_bytes);
    const uint32_t out_row_stride = get_arg(args::out_row_stride);

    // The plan's count, not get_num_threads(): were a hart not to run, its rows would be missing.
    constexpr uint32_t R = get_arg(args::reader_threads);
    const uint32_t t = get_my_thread_id();
    constexpr NocOptVals user_txn = {.trid = 0};

    Noc noc;
    const auto input = TensorAccessor(tensor::input);
    const LocalTensorAccessor<uint8_t> out_shard(tensor::output);

    for (uint32_t block_start = 0; block_start < num_rows; block_start += block_rows) {
        const uint32_t rows_this_block = (num_rows - block_start < block_rows) ? num_rows - block_start : block_rows;

        // gather_rows_block: this hart's contiguous sub-range of the block.
        const uint32_t q = rows_this_block / R;
        const uint32_t rem = rows_this_block % R;
        const uint32_t lo = block_start + t * q + (t < rem ? t : rem);
        const uint32_t count = q + (t < rem ? 1 : 0);
        if (count == 0) {
            continue;
        }

        // (n, h, w) of the hart's first row, once; then counters.
        const uint32_t row = first_row + lo;
        uint32_t w_i = row % block_w;
        const uint32_t nh = row / block_w;
        uint32_t h_i = nh % block_h;
        uint32_t n_i = nh / block_h;
        uint32_t page_id = ((start_n + n_i) * in_h + start_h + h_i) * in_w + start_w + w_i;
        uint32_t dst_offset = lo * out_row_stride;

        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read<NocOptions::TXN_ID>(
                input,
                out_shard,
                row_bytes,
                {.page_id = page_id, .offset_bytes = src_offset_bytes},
                {.offset_bytes = dst_offset},
                user_txn);
            dst_offset += out_row_stride;
            ++page_id;
            if (++w_i == block_w) {
                // End of a w-run: the next row starts a new (n, h) image row.
                w_i = 0;
                if (++h_i == block_h) {
                    h_i = 0;
                    ++n_i;
                }
                page_id = ((start_n + n_i) * in_h + start_h + h_i) * in_w + start_w;
            }
        }
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
    }

    // Hand the hart back with the user id in its tag register (target page, ring protocols).
    noc_async_read_set_trid(0, noc.get_noc_id());
}
