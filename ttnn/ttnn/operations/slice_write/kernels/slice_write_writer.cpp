// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// slice_write writer: store_block. Writes every valid row of this core's shard to its destination page.
//
// The page map (op_design.md, Blocking Model): source row r of the block, (n, h, w) =
// (r / (h_s*w_s), (r / w_s) mod h_s, r mod w_s), goes to destination page
// ((n + start_n)*H + h + start_h)*W + w + start_w. It is walked incrementally: one division at the first row
// of an entry, then additions.
//
// Writer hart t of W owns entries t, t+W, ... of the ring over the shard; entry e holds shard rows
// [e*entry_rows, (e+1)*entry_rows). A ragged last shard keeps the nominal wait/pop of one entry and writes
// only its valid rows.
//
// Explicit sync on a borrowed ring: every NoC transfer of this kernel carries user transaction id 0 (and the
// barrier waits on it), and the hart's tag is handed back as the last statement, so neither an earlier
// implicit-sync program nor a later untagged one sees a stale ring tag (target page, ring protocols).
//
// Raw NoC API by design: the step is "write N rows of a resident ring entry to N accessor pages", one raw
// write per row (each row is its own destination page, in its own bank). The dataflow kernel library has no
// L1->DRAM page-write helper (op_design.md, Helpers considered and rejected).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t src_shard_entries = get_arg(args::src_shard_entries);
    constexpr uint32_t rows_per_write_batch = get_arg(args::rows_per_write_batch);
    constexpr NocOptVals user_txn = {.trid = 0};

    const uint32_t first_row = get_arg(args::first_row);  // this core's first source row (shard order)
    const uint32_t num_rows = get_arg(args::num_rows);    // valid rows in this core's shard
    const uint32_t row_bytes = get_arg(args::row_bytes);
    const uint32_t src_row_stride_bytes = get_arg(args::src_row_stride_bytes);
    const uint32_t entry_rows = get_arg(args::entry_rows);
    const uint32_t block_h = get_arg(args::block_h);
    const uint32_t block_w = get_arg(args::block_w);
    const uint32_t dst_h = get_arg(args::dst_h);
    const uint32_t dst_w = get_arg(args::dst_w);
    const uint32_t start_n = get_arg(args::start_n);
    const uint32_t start_h = get_arg(args::start_h);
    const uint32_t start_w = get_arg(args::start_w);

    // Page jumps of the walk: end of a run of block_w pages, end of a plane of block_h runs.
    const uint32_t row_jump = dst_w - block_w;
    const uint32_t plane_jump = (dst_h - block_h) * dst_w;

    DataflowBuffer dfb_src_shard(dfb::src_shard);
    Noc noc;
    const auto dst = TensorAccessor(tensor::dst);

    for (uint32_t entry = get_my_thread_id(); entry < src_shard_entries; entry += W) {
        const uint32_t entry_first = entry * entry_rows;
        const uint32_t rows = num_rows > entry_first ? (num_rows - entry_first < entry_rows ? num_rows - entry_first
                                                                                              : entry_rows)
                                                     : 0;
        dfb_src_shard.wait_front(1);
        if (rows > 0) {
            // Position of the entry's first row in the block, and its destination page.
            const uint32_t r0 = first_row + entry_first;
            const uint32_t plane = block_h * block_w;
            const uint32_t n = r0 / plane;
            const uint32_t in_plane = r0 - n * plane;
            uint32_t h = in_plane / block_w;
            uint32_t w = in_plane - h * block_w;
            uint32_t dst_page = ((n + start_n) * dst_h + h + start_h) * dst_w + w + start_w;

            uint32_t src_offset = 0;
            uint32_t in_batch = 0;
            for (uint32_t i = 0; i < rows; ++i) {
                noc.async_write<NocOptions::TXN_ID>(
                    dfb_src_shard, dst, row_bytes, {.offset_bytes = src_offset}, {.page_id = dst_page}, user_txn);
                if (++in_batch == rows_per_write_batch) {
                    noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                    in_batch = 0;
                }
                src_offset += src_row_stride_bytes;
                ++dst_page;
                if (++w == block_w) {
                    w = 0;
                    dst_page += row_jump;
                    if (++h == block_h) {
                        h = 0;
                        dst_page += plane_jump;
                    }
                }
            }
            noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
        }
        dfb_src_shard.pop_front(1);
    }
    // Hand the hart back with the user id in its tag register.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
