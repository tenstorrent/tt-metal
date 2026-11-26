// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rows_tile writer (BRISC, NoC1): regroup_block + store_block.
//
// Row move (n, a, b, wt): input tile (n, a, b/32, wt) row b%32 -> output tile (n, b, a/32, wt) row a%32.
//
// Regroup path (block_a > 1): each live row is copied input slot -> staging slot with a self-aimed 32 B NoC read
// (dataflow_kernel_lib::read_with_state), then the staging tiles go to DRAM whole (a-span covers the tile's live
// rows) or as one run per face column. cb_output_staging is private scratch: no CB handshake; slot k % depth is
// reused only after the flush that retires block k - depth's writes.
//
// Direct path (block_a == 1): every row is written from its input slot straight to its output position, always as
// face-row runs: the slot holds rows at input positions, so a whole-tile write would put them in the wrong rows.
//
// Store uses raw NoC + TensorAccessor: no kernel_lib dataflow helper writes interleaved DRAM pages or sub-pages.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/ttnn/operations/g6_permute/kernels/g6_permute_common.hpp"

using namespace g6_permute;

namespace {

template <typename Accessor>
FORCE_INLINE void store_block_direct(const Accessor& output, const Block& blk, uint32_t in_base) {
    const uint32_t at = blk.a0 / tile_height;
    const uint32_t ra = blk.a0 % tile_height;
    for (uint32_t wi = 0; wi < blk.live_wt; ++wi) {
        const uint32_t slot = in_base + wi * tile_bytes;
        for (uint32_t bi = 0; bi < blk.live_b; ++bi) {
            const uint32_t b = blk.b0 + bi;
            const uint32_t rb = b % tile_height;
            const uint32_t page = out_page(blk.n, b, at, blk.wt0 + wi);
            for (uint32_t h = 0; h < face_cols; ++h) {
                noc_async_write(slot + face_off(rb, h), output.get_noc_addr(page, face_off(ra, h)), face_row_bytes);
            }
        }
    }
}

FORCE_INLINE void regroup_block(Noc noc, const Block& blk, uint32_t in_base, uint32_t staging_base) {
    const uint32_t ra0 = blk.a0 % tile_height;
    for (uint32_t ai = 0; ai < blk.live_a; ++ai) {
        const uint32_t ra = ra0 + ai;
        for (uint32_t wi = 0; wi < blk.live_wt; ++wi) {
            const uint32_t src_slot = in_base + (ai * block_wt + wi) * tile_bytes;
            for (uint32_t bi = 0; bi < blk.live_b; ++bi) {
                const uint32_t rb = (blk.b0 + bi) % tile_height;
                const uint32_t dst_slot = staging_base + (bi * block_wt + wi) * tile_bytes;
                for (uint32_t h = 0; h < face_cols; ++h) {
                    dataflow_kernel_lib::read_with_state<face_row_bytes>(
                        noc, dst_slot + face_off(ra, h), src_slot + face_off(rb, h));
                }
            }
        }
    }
}

template <typename Accessor>
FORCE_INLINE void store_block_staged(const Accessor& output, const Block& blk, uint32_t staging_base) {
    const uint32_t at = blk.a0 / tile_height;
    const uint32_t ra0 = blk.a0 % tile_height;
    const bool whole_tile = ra0 == 0 && blk.live_a == live_rows_of_tile_row(A, at);
    for (uint32_t bi = 0; bi < blk.live_b; ++bi) {
        for (uint32_t wi = 0; wi < blk.live_wt; ++wi) {
            const uint32_t page = out_page(blk.n, blk.b0 + bi, at, blk.wt0 + wi);
            const uint32_t slot = staging_base + (bi * block_wt + wi) * tile_bytes;
            if (whole_tile) {
                noc_async_write(slot, output.get_noc_addr(page), tile_bytes);
            } else {
                for_each_row_run(ra0, blk.live_a, [&](uint32_t offset, uint32_t bytes) {
                    noc_async_write(slot + offset, output.get_noc_addr(page, offset), bytes);
                });
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<num_plan_ct_args>();

    const uint32_t start_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);

    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);
    Noc noc;

    uint32_t staging_region = 0;
    if constexpr (regroup) {
        staging_region = get_write_ptr(cb_output_staging);
        // The state holds this core's NoC coordinates and the copy size; each copy rewrites both L1 addresses.
        dataflow_kernel_lib::set_read_state<face_row_bytes>(noc, staging_region);
    }

    for (uint32_t k = 0; k < num_blocks; ++k) {
        const Block blk = decode_block(start_block + k);
        {
            DeviceZoneScopedN("wait_input");
            cb_wait_front(cb_input_tiles, input_pages_per_block);
        }
        const uint32_t in_base = get_read_ptr(cb_input_tiles);

        if constexpr (regroup) {
            const uint32_t staging_base = staging_region + (k % depth_staging) * staging_pages_per_block * tile_bytes;
            if constexpr (depth_staging == 1) {
                if (k > 0) {
                    DeviceZoneScopedN("staging_flush");
                    noc_async_writes_flushed();
                }
            }
            {
                DeviceZoneScopedN("regroup_block");
                regroup_block(noc, blk, in_base, staging_base);
            }
            {
                DeviceZoneScopedN("regroup_barrier");
                noc_async_read_barrier();
            }
            cb_pop_front(cb_input_tiles, input_pages_per_block);
            {
                // Retires block k-1's writes, which drained during this regroup; with depth >= 2 that frees the
                // slot block k+1 regroups into.
                DeviceZoneScopedN("store_flush");
                noc_async_writes_flushed();
            }
            {
                DeviceZoneScopedN("store_block");
                store_block_staged(output, blk, staging_base);
            }
        } else {
            {
                DeviceZoneScopedN("store_block");
                store_block_direct(output, blk, in_base);
            }
            {
                // The writes read the input slot: it must leave L1 before the reader may refill it.
                DeviceZoneScopedN("store_flush");
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_input_tiles, input_pages_per_block);
        }
    }

    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
