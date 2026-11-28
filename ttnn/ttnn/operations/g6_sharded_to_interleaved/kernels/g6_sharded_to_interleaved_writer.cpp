// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_sharded_to_interleaved — owner-writes store kernel (one write lane).
//
// Regime "owner_writes": the core holding shard k writes every tile of its resident shard
// straight from L1 to the tile's interleaved output page. Each tile crosses the NoC exactly once.
// The shard sits in cb_input_shard, a CB globally allocated on the input's shard buffer
// (zero-copy) — it is never read through a TensorAccessor, only via its read pointer.
//
// No kernel_lib helper applies: every kernel_lib helper is a compute helper (unpack -> DEST -> pack);
// this op has no compute phase and must copy bfloat8_b bytes (shared exponents) verbatim. The
// store is the raw dataflow NoC API (noc_async_write_page + TensorAccessor), per the design.
//
// Lane (sub-rectangle) of the shard handled by this kernel instance:
//   rows [0, total_rows) x cols [0, block_cols), slot origin src_first_slot, page origin first_page.
// The lane is walked in num_blocks blocks of up to block_rows rows each (Phase 0: one block = the
// whole lane). Slot (r, c) -> page first_page + r * tensor_col_tiles + c; the slot row stride is
// the FULL shard width (shard_col_tiles), even for a short last width shard (padding slots).
//
// bank_stagger: the lane's row-major slot walk starts at linear slot start_slot (host-derived so
// lanes whose first pages share an output bank start on different banks) and wraps; every slot is
// still written exactly once. start_slot = 0 is the plain in-order walk.
//
// dual_noc (CT): the kernel runs in DM_DYNAMIC_NOC mode and alternates its writes between its own
// NoC and the other one, so one RISC injects into both NoCs' links; both NoCs are drained at the end.
// dual_noc = 0 issues every write on noc_index (the kernel's configured NoC).
//
// CT args: cb_input_shard, shard_col_tiles, tensor_col_tiles, out_page_size, in_slot_bytes,
//          dual_noc, TensorAccessorArgs(output)
// Common RT args: out_addr
// RT args: num_blocks, block_rows, block_cols, total_rows, first_page, src_first_slot, start_slot

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_shard = get_compile_time_arg_val(0);
    constexpr uint32_t shard_col_tiles = get_compile_time_arg_val(1);   // slot row stride
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(2);  // page row stride
    constexpr uint32_t out_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t in_slot_bytes = get_compile_time_arg_val(4);  // input aligned page size
    constexpr bool dual_noc = get_compile_time_arg_val(5) != 0;
    constexpr auto out_args = TensorAccessorArgs<6>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t block_rows = get_arg_val<uint32_t>(1);
    const uint32_t block_cols = get_arg_val<uint32_t>(2);
    const uint32_t total_rows = get_arg_val<uint32_t>(3);
    const uint32_t first_page = get_arg_val<uint32_t>(4);
    const uint32_t src_first_slot = get_arg_val<uint32_t>(5);
    const uint32_t start_slot = get_arg_val<uint32_t>(6);

    if (num_blocks == 0) {
        return;  // grid core with no shard (k >= num_shards) or an empty lane
    }

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_size);
    const uint32_t shard_base = get_read_ptr(cb_input_shard) + src_first_slot * in_slot_bytes;

    constexpr uint32_t src_row_bytes = shard_col_tiles * in_slot_bytes;
    // Rotated walk position (r, c); one division per lane, not per slot.
    uint32_t r = start_slot / block_cols;
    uint32_t c = start_slot - r * block_cols;
    uint32_t src_row = shard_base + r * src_row_bytes;
    uint32_t page_row = first_page + r * tensor_col_tiles;
    uint8_t noc = noc_index;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("S2I_STORE_BLOCK");
        // store_block: the next `rows` rows' worth of slots of the (rotated) lane walk.
        const uint32_t rows_left = total_rows - block_idx * block_rows;
        const uint32_t rows = rows_left < block_rows ? rows_left : block_rows;
        for (uint32_t n = rows * block_cols; n > 0; --n) {
            noc_async_write_page(page_row + c, out_accessor, src_row + c * in_slot_bytes, out_page_size, 0, noc);
            if constexpr (dual_noc) {
                noc ^= 1;
            }
            if (++c == block_cols) {
                c = 0;
                if (++r == total_rows) {
                    r = 0;
                    src_row = shard_base;
                    page_row = first_page;
                } else {
                    src_row += src_row_bytes;
                    page_row += tensor_col_tiles;
                }
            }
        }
    }

    {
        DeviceZoneScopedN("S2I_DRAIN");
        noc_async_write_barrier(noc_index);  // BB-03: every write acked before return
        if constexpr (dual_noc) {
            noc_async_write_barrier(1 - noc_index);
        }
    }
}
