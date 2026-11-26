// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rows_tile reader (NCRISC, NoC0): load_block.
//
// Raw NoC + TensorAccessor instead of a kernel_lib helper: no dataflow helper issues interleaved DRAM page or
// sub-page transfers (local_copy_helpers only address this core's own L1).
//
// Per block: the block_a x block_wt input tiles of one b tile row land in whole-tile slots of cb_input_tiles.
// A tile is read whole when the block's b-span covers all its live rows; otherwise only the span's rows, as one
// run per face column, landing at the same tile-relative offset as the source: BH DRAM reads require local and
// NoC addresses equal mod 64, which a compacted landing slot would break for odd rows.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/ttnn/operations/g6_permute/kernels/g6_permute_common.hpp"

using namespace g6_permute;

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<num_plan_ct_args>();

    const uint32_t start_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);

    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);

    for (uint32_t k = 0; k < num_blocks; ++k) {
        const Block blk = decode_block(start_block + k);
        const uint32_t bt = blk.b0 / tile_height;
        const uint32_t rb0 = blk.b0 % tile_height;
        const bool whole_tile = rb0 == 0 && blk.live_b == live_rows_of_tile_row(B, bt);

        {
            DeviceZoneScopedN("load_block_reserve");
            cb_reserve_back(cb_input_tiles, input_pages_per_block);
        }
        const uint32_t block_base = get_write_ptr(cb_input_tiles);
        {
            DeviceZoneScopedN("load_block_issue");
            for (uint32_t ai = 0; ai < blk.live_a; ++ai) {
                for (uint32_t wi = 0; wi < blk.live_wt; ++wi) {
                    const uint32_t page = in_page(blk.n, blk.a0 + ai, bt, blk.wt0 + wi);
                    const uint32_t slot = block_base + (ai * block_wt + wi) * tile_bytes;
                    if (whole_tile) {
                        noc_async_read(input.get_noc_addr(page), slot, tile_bytes);
                    } else {
                        for_each_row_run(rb0, blk.live_b, [&](uint32_t offset, uint32_t bytes) {
                            noc_async_read(input.get_noc_addr(page, offset), slot + offset, bytes);
                        });
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("load_block_barrier");
            noc_async_read_barrier();
        }
        // Nominal page count on ragged blocks too: the ring closes only on whole quanta.
        cb_push_back(cb_input_tiles, input_pages_per_block);
    }
}
