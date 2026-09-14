// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (BRISC).
//
// Per block (Cw > 1 builds only — the cross-core face of the W axis):
//   exchange_partials_block  peers unicast their `rows` collapsed partials into the root's cb_gather
//                            slot (r*Cw + w_split_index) and bump the root's SEM_GATHER; the root
//                            copies its own partial into its slot, waits for Cw-1 arrivals, resets
//                            the semaphore and publishes the round.
//   broadcast_rstd_block     root: SenderPipe::send(cb_rstd_handoff -> cb_rstd) with loopback (the
//                            root's own copy lands in its cb_rstd); every other core in the group
//                            rectangle (incl. passive bbox cores in R3) ReceiverPipe::receive()s.
// Per block (interleaved): store_block — all `rows * Wc` tile writes issued, ONE barrier, ONE pop
//                          (RM: write_sticks_after_untilize, one barrier per tile-row). R3: nothing —
//                          the output shard is the tensor itself.
//
// Slot addresses are base-relative: cb_gather / cb_rstd hold exactly one round, so every round's
// reserve returns the CB base and the address is identical on every core (uniform, first-created
// CB descriptors). Semaphore-reset ordering: a peer sends partial k+1 only after consuming rstd_k,
// which the root multicasts strictly after wait(num_partials) + set(0) of round k.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    // ---- named compile-time args ----
    constexpr uint32_t cb_output_tiles = get_named_compile_time_arg_val("CB_OUTPUT_TILES");
    constexpr uint32_t cb_out_sticks = get_named_compile_time_arg_val("CB_OUT_STICKS");
    constexpr uint32_t cb_partial_collapsed = get_named_compile_time_arg_val("CB_PARTIAL_COLLAPSED");
    constexpr uint32_t cb_gather = get_named_compile_time_arg_val("CB_GATHER");
    constexpr uint32_t cb_rstd_handoff = get_named_compile_time_arg_val("CB_RSTD_HANDOFF");
    constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("CB_RSTD");
    constexpr bool output_rm = get_named_compile_time_arg_val("INPUT_RM") != 0;  // output layout == input layout
    constexpr bool sharded = get_named_compile_time_arg_val("SHARDED") != 0;
    constexpr uint32_t num_w_splits = get_named_compile_time_arg_val("NUM_W_SPLITS");
    constexpr uint32_t sem_gather = get_named_compile_time_arg_val("SEM_GATHER");
    constexpr uint32_t out_page_bytes = get_named_compile_time_arg_val("OUT_PAGE_BYTES");  // tile (TILE) or stick (RM)
    constexpr uint32_t out_tile_bytes = get_named_compile_time_arg_val("OUT_TILE_BYTES");
    constexpr uint32_t out_elem_bytes = get_named_compile_time_arg_val("OUT_ELEM_BYTES");
    constexpr uint32_t partial_tile_bytes = get_named_compile_time_arg_val("P32_BYTES");
    constexpr uint32_t mcast_ct_base = get_named_compile_time_arg_val("MCAST_CT_BASE");
    constexpr uint32_t mcast_rt_base = get_named_compile_time_arg_val("MCAST_RT_BASE");
    constexpr uint32_t tile_rows = 32;

    // ---- positional compile-time args: [mcast CT block (6)] [output TensorAccessorArgs] ----
    constexpr auto output_args = TensorAccessorArgs<mcast_ct_base + 6>();

    // ---- runtime args ----
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_rows = get_arg_val<uint32_t>(3);
    const uint32_t last_block_rows = get_arg_val<uint32_t>(4);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(6);
    const uint32_t tensor_w_tiles = get_arg_val<uint32_t>(7);
    const bool is_active = get_arg_val<uint32_t>(8) != 0;
    const bool is_root = get_arg_val<uint32_t>(9) != 0;
    const uint32_t w_split_index = get_arg_val<uint32_t>(10);
    const uint32_t root_x = get_arg_val<uint32_t>(11);
    const uint32_t root_y = get_arg_val<uint32_t>(12);
    const uint32_t num_partials_expected = get_arg_val<uint32_t>(13);

    const auto output_acc = TensorAccessor(output_args, output_addr, out_page_bytes);

    // store_block: this core's rows*Wc output tiles (or 32*rows stick chunks) -> DRAM. R3: nothing.
    const auto store_block = [&](uint32_t block_row_tile_start, uint32_t rows) {
        if constexpr (sharded) {
            return;
        }
        if (!is_active) {
            return;
        }
        if constexpr (output_rm) {
            write_sticks_after_untilize<cb_out_sticks>(
                output_acc,
                tile_rows * rows,
                core_w_tiles * tile_rows * out_elem_bytes,
                tile_rows * block_row_tile_start,
                w_tile_start * tile_rows * out_elem_bytes);
        } else {
            const uint32_t block_tiles = rows * core_w_tiles;
            cb_wait_front(cb_output_tiles, block_tiles);
            uint32_t src = get_read_ptr(cb_output_tiles);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row_base = (block_row_tile_start + r) * tensor_w_tiles + w_tile_start;
                for (uint32_t c = 0; c < core_w_tiles; ++c) {
                    noc_async_write(src, output_acc.get_noc_addr(row_base + c), out_tile_bytes);
                    src += out_tile_bytes;
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, block_tiles);
        }
    };

    if constexpr (num_w_splits > 1) {
        Noc noc;
        Semaphore<> gather_sem(sem_gather);
        constexpr auto mcast = McastArgs<mcast_ct_base, mcast_rt_base>();
        auto sender = mcast.sender(noc);
        auto receiver = mcast.receiver(noc);
        // Peers address the root's cb_gather through their own base: the CB is uniform and created
        // first, so the address is identical on every core and never moves (one-round capacity).
        const uint32_t gather_base = get_write_ptr(cb_gather);

        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
            const uint32_t block_row_tile_start = row_tile_start + block_idx * block_rows;

            // exchange_partials_block
            if (is_active) {
                cb_wait_front(cb_partial_collapsed, rows);
                const uint32_t src = get_read_ptr(cb_partial_collapsed);
                if (is_root) {
                    cb_reserve_back(cb_gather, rows * num_w_splits);
                    // own slot: local L1 -> L1 copy through the NoC (strided destination)
                    for (uint32_t r = 0; r < rows; ++r) {
                        noc_async_read(
                            get_noc_addr(my_x[noc_index], my_y[noc_index], src + r * partial_tile_bytes),
                            gather_base + (r * num_w_splits + w_split_index) * partial_tile_bytes,
                            partial_tile_bytes);
                    }
                    noc_async_read_barrier();
                    gather_sem.wait(num_partials_expected);
                    gather_sem.set(0);
                    cb_push_back(cb_gather, rows * num_w_splits);
                } else {
                    for (uint32_t r = 0; r < rows; ++r) {
                        noc_async_write(
                            src + r * partial_tile_bytes,
                            get_noc_addr(
                                root_x, root_y, gather_base + (r * num_w_splits + w_split_index) * partial_tile_bytes),
                            partial_tile_bytes);
                    }
                    noc_async_write_barrier();
                    gather_sem.up(noc, root_x, root_y, 1);
                }
                cb_pop_front(cb_partial_collapsed, rows);
            }

            // broadcast_rstd_block
            if (is_root) {
                cb_wait_front(cb_rstd_handoff, rows);
                cb_reserve_back(cb_rstd, rows);
                sender.send(get_read_ptr(cb_rstd_handoff), get_write_ptr(cb_rstd), rows * partial_tile_bytes);
                cb_push_back(cb_rstd, rows);
                cb_pop_front(cb_rstd_handoff, rows);
            } else {
                if (is_active) {
                    cb_reserve_back(cb_rstd, rows);
                }
                receiver.receive();
                if (is_active) {
                    cb_push_back(cb_rstd, rows);
                }
            }

            store_block(block_row_tile_start, rows);
        }
        noc_async_atomic_barrier();  // the peers' semaphore increments have all completed
    } else {
        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
            store_block(row_tile_start + block_idx * block_rows, rows);
        }
    }
}
