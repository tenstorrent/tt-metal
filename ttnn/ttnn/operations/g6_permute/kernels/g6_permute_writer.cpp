// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rows_tile writer: regroup_block + store_block.
//
// block_a > 1 (regroup path): every live row move (a, b) is a 32-B self-read per face half from the
// input slot of a (row b % 32) to the staging tile of b (row a % 32); the staged output tiles then go
// to DRAM whole, or as one run per face half when the block's a-span does not cover the tile's live rows.
// block_a == 1 (direct path): the input slot already holds each output row contiguously, so every row
// goes straight to its output position; never as a whole tile, because the slot's rows sit at input
// positions.
//
// regroup_block uses the kernel_lib local-copy helpers; the DRAM stores are raw NoC + TensorAccessor
// because no dataflow helper issues interleaved DRAM page or sub-page writes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_permute_common.hpp"

using namespace g6_permute;

constexpr uint32_t CB_OUTPUT_STAGING = get_compile_time_arg_val(NUM_SHARED_CT_ARGS);
constexpr uint32_t DEPTH_STAGING = get_compile_time_arg_val(NUM_SHARED_CT_ARGS + 1);
constexpr uint32_t STAGING_PAGES_PER_BLOCK = BLOCK_B * BLOCK_WT;
constexpr bool REGROUP = BLOCK_A > 1;

static_assert(!REGROUP || DEPTH_STAGING >= 1, "the regroup path needs a staging buffer");

FORCE_INLINE uint32_t wait_input_block() {
    {
        DeviceZoneScopedN("wait_input_block");
        cb_wait_front(CB_INPUT_TILES, INPUT_PAGES_PER_BLOCK);
    }
    return get_read_ptr(CB_INPUT_TILES);
}

FORCE_INLINE void regroup_block(Noc noc, const Block& blk, uint32_t stage) {
    const uint32_t slots = wait_input_block();
    if constexpr (DEPTH_STAGING == 1) {
        // One staging slot: the previous block's writes still read it.
        DeviceZoneScopedN("regroup_block_flush");
        noc_async_writes_flushed();
    }
    const uint32_t ra0 = blk.a0 % TILE_H;
    const uint32_t rb0 = blk.b0 % TILE_H;
    {
        DeviceZoneScopedN("regroup_block_issue");
        for (uint32_t a = 0; a < blk.live_a; ++a) {
            for (uint32_t wt = 0; wt < blk.live_wt; ++wt) {
                const uint32_t src_tile = slots + (a * BLOCK_WT + wt) * TILE_BYTES;
                for (uint32_t b = 0; b < blk.live_b; ++b) {
                    const uint32_t dst_tile = stage + (b * BLOCK_WT + wt) * TILE_BYTES;
                    for (uint32_t h = 0; h < FACES_PER_BAND; ++h) {
                        dataflow_kernel_lib::read_with_state<FACE_ROW_BYTES>(
                            noc, dst_tile + face_off(ra0 + a, h), src_tile + face_off(rb0 + b, h));
                    }
                }
            }
        }
    }
    {
        DeviceZoneScopedN("regroup_block_read_barrier");
        noc_async_read_barrier();
    }
    cb_pop_front(CB_INPUT_TILES, INPUT_PAGES_PER_BLOCK);
}

template <typename Accessor>
FORCE_INLINE void store_staged_block(const Accessor& output, const Block& blk, uint32_t stage) {
    {
        // Retires the previous block's writes, which drained during this block's regroup; with two
        // staging slots that frees the slot the next block regroups into.
        DeviceZoneScopedN("store_block_flush");
        noc_async_writes_flushed();
    }
    const uint32_t at = blk.a0 / TILE_H;
    const uint32_t ra0 = blk.a0 % TILE_H;
    const bool whole = covers_live_rows(ra0, blk.live_a, A - at * TILE_H);
    DeviceZoneScopedN("store_block_issue");
    for (uint32_t b = 0; b < blk.live_b; ++b) {
        for (uint32_t wt = 0; wt < blk.live_wt; ++wt) {
            const uint32_t page = out_page(blk.n, blk.b0 + b, at, blk.wt0 + wt);
            const uint32_t src_tile = stage + (b * BLOCK_WT + wt) * TILE_BYTES;
            if (whole) {
                noc_async_write<TILE_BYTES>(src_tile, output.get_noc_addr(page), TILE_BYTES);
            } else {
                for_each_row_run(ra0, blk.live_a, [&](uint32_t offset, uint32_t bytes) {
                    noc_async_write<TILE_BYTES>(src_tile + offset, output.get_noc_addr(page, offset), bytes);
                });
            }
        }
    }
}

template <typename Accessor>
FORCE_INLINE void store_block_direct(const Accessor& output, const Block& blk) {
    const uint32_t slots = wait_input_block();
    const uint32_t at = blk.a0 / TILE_H;
    const uint32_t ra0 = blk.a0 % TILE_H;
    const uint32_t rb0 = blk.b0 % TILE_H;
    {
        DeviceZoneScopedN("store_block_issue");
        for (uint32_t wt = 0; wt < blk.live_wt; ++wt) {
            const uint32_t src_tile = slots + wt * TILE_BYTES;
            for (uint32_t b = 0; b < blk.live_b; ++b) {
                const uint32_t page = out_page(blk.n, blk.b0 + b, at, blk.wt0 + wt);
                for (uint32_t h = 0; h < FACES_PER_BAND; ++h) {
                    noc_async_write<FACE_ROW_BYTES>(
                        src_tile + face_off(rb0 + b, h), output.get_noc_addr(page, face_off(ra0, h)), FACE_ROW_BYTES);
                }
            }
        }
    }
    {
        // The writes read the input slot; the reader may refill it only after they left L1.
        DeviceZoneScopedN("store_block_flush");
        noc_async_writes_flushed();
    }
    cb_pop_front(CB_INPUT_TILES, INPUT_PAGES_PER_BLOCK);
}

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<NUM_SHARED_CT_ARGS + 2>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr, TILE_BYTES);

    if constexpr (REGROUP) {
        Noc noc;
        // Private scratch: the writer is its only producer and consumer, so it never takes part in a
        // CB handshake; slot reuse is ordered by the write flushes instead.
        const uint32_t staging_base = get_write_ptr(CB_OUTPUT_STAGING);
        // The state holds the NoC coordinates (this core) and the copy size; every read rewrites
        // both addresses, so one programming serves every row move of the kernel.
        dataflow_kernel_lib::set_read_state<FACE_ROW_BYTES>(noc, staging_base);
        uint32_t stage_slot = 0;
        for (uint32_t k = 0; k < num_blocks; ++k) {
            const Block blk = decode_block(start_block + k);
            const uint32_t stage = staging_base + stage_slot * STAGING_PAGES_PER_BLOCK * TILE_BYTES;
            regroup_block(noc, blk, stage);
            store_staged_block(output, blk, stage);
            stage_slot = (stage_slot + 1 == DEPTH_STAGING) ? 0 : stage_slot + 1;
        }
    } else {
        for (uint32_t k = 0; k < num_blocks; ++k) {
            store_block_direct(output, decode_block(start_block + k));
        }
    }

    DeviceZoneScopedN("store_write_barrier");
    noc_async_write_barrier();
}
