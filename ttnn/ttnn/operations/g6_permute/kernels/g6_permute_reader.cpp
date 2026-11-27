// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rows_tile reader (load_block): brings the block's input tiles into cb_input_tiles, one slot per
// (a, wt). A tile whose live rows the block's b-span covers moves as one page; otherwise only the
// span's rows move, as one run per face half.
//
// Raw NoC + TensorAccessor instead of a kernel_lib helper: no dataflow helper issues interleaved
// DRAM page or sub-page reads (op_design.md, "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

using namespace g6_permute;

template <typename Accessor>
FORCE_INLINE void load_block(const Accessor& input, const Block& blk) {
    {
        DeviceZoneScopedN("load_block_reserve");
        cb_reserve_back(CB_INPUT_TILES, INPUT_PAGES_PER_BLOCK);
    }
    const uint32_t slots = get_write_ptr(CB_INPUT_TILES);
    const uint32_t bt = blk.b0 / TILE_H;
    const uint32_t rb0 = blk.b0 % TILE_H;
    const bool whole = covers_live_rows(rb0, blk.live_b, B - bt * TILE_H);
    {
        DeviceZoneScopedN("load_block_issue");
        for (uint32_t a = 0; a < blk.live_a; ++a) {
            for (uint32_t wt = 0; wt < blk.live_wt; ++wt) {
                const uint32_t page = in_page(blk.n, blk.a0 + a, bt, blk.wt0 + wt);
                const uint32_t slot = slots + (a * BLOCK_WT + wt) * TILE_BYTES;
                if (whole) {
                    noc_async_read<TILE_BYTES>(input.get_noc_addr(page), slot, TILE_BYTES);
                } else {
                    // A run lands at its source's tile-relative offset: BH DRAM reads need source and
                    // destination equal mod 64, and both bases are aligned beyond that.
                    for_each_row_run(rb0, blk.live_b, [&](uint32_t offset, uint32_t bytes) {
                        noc_async_read<TILE_BYTES>(input.get_noc_addr(page, offset), slot + offset, bytes);
                    });
                }
            }
        }
    }
    {
        DeviceZoneScopedN("load_block_read_barrier");
        noc_async_read_barrier();
    }
    cb_push_back(CB_INPUT_TILES, INPUT_PAGES_PER_BLOCK);
}

void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<NUM_SHARED_CT_ARGS>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr, TILE_BYTES);

    for (uint32_t k = 0; k < num_blocks; ++k) {
        load_block(input, decode_block(start_block + k));
    }
}
