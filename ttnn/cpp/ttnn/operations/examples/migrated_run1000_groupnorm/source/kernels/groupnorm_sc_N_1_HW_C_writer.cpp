// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C writer.
//
// Drains cb_output_tiles in the SAME channel-major order the compute kernel
// packs it (channel-tile `t` outer, HW-row `r` inner) and writes whole tile
// pages back to the (N,1,HW,C) tiled output through a TensorAccessor.
//
// Work unit = (batch n, cluster k); the HW axis is walked in blocks of
// BLOCK_HW_TILES, the live block knob shared with the reader and compute.
//
// Refinement 3 — in REGIME_HW_SPLIT the writer ALSO owns the cross-core moment
// combine, because it is idle for the whole moment pass. Per unit the row of
// `hw_split_factor` cores does:
//
//   every non-combiner : NoC-write its (Sigma-x, Sigma-x^2) partial pair into
//                        its own slot of the combiner's cb_partial_moments,
//                        then bump the combiner's SEM_GATHER;
//   the combiner (col 0): wait SEM_GATHER == S-1, publish cb_partial_moments to
//                        its compute kernel, take back the summed totals in
//                        cb_moment_total_out, and MULTICAST them down the row
//                        (loopback, so its own copy lands too);
//   every core          : publish the landed totals as cb_moment_total_in.
//
// The apply pass then runs locally and unchanged on each core's own HW slice.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 1;
constexpr uint32_t cb_moment_sum = 4;
constexpr uint32_t cb_moment_sq = 5;
constexpr uint32_t cb_partial_moments = 17;
constexpr uint32_t cb_moment_total_out = 18;
constexpr uint32_t cb_moment_total_in = 19;

// --- geometry (shared geom_ct prefix) ---------------------------------------
constexpr uint32_t tensor_hw_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cluster_c_tiles = get_compile_time_arg_val(2);
// (3) groups_per_cluster, (4) num_clusters used below
constexpr uint32_t num_clusters = get_compile_time_arg_val(4);
constexpr uint32_t BLOCK_HW_TILES = get_compile_time_arg_val(5);
constexpr uint32_t num_hw_blocks = get_compile_time_arg_val(6);
// (7..12) Cg / cluster_channels / mask geometry / ragged tails — compute+reader only.
constexpr uint32_t regime_id = get_compile_time_arg_val(13);
constexpr uint32_t hw_split_factor = get_compile_time_arg_val(14);

constexpr bool HW_SPLIT = (regime_id == 1);
constexpr uint32_t n_partial_tiles = HW_SPLIT ? (hw_split_factor - 1) * 2 * cluster_c_tiles : 0;
constexpr uint32_t n_total_tiles = 2 * cluster_c_tiles;

// The mcast helper's CT/RT blocks sit between the geometry prefix and the output
// accessor so both decoders keep a fixed base.
constexpr auto mcast = dataflow_kernel_lib::McastArgs</*CT=*/16, /*RT=*/6>();
constexpr auto output_args = TensorAccessorArgs<22>();

// Gather counter the combiner waits on (SEM_GATHER on the host side).
constexpr uint32_t SEM_GATHER = 1;

// Bound outstanding NoC writes so the barrier does not serialize per page.
constexpr uint32_t writes_per_barrier = 8;
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    const uint32_t hw_tile_start = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const bool is_combiner = get_arg_val<uint32_t>(4) != 0;
    [[maybe_unused]] const uint32_t gather_slot = get_arg_val<uint32_t>(5);

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto out = TensorAccessor(output_args, out_addr, tile_bytes);

    if constexpr (HW_SPLIT) {
        // Both pipes are constructed HERE, before this core contributes anything
        // to the gather. ReceiverPipe's ctor resets the data-ready flag, so it
        // must run strictly before the combiner can possibly signal — and the
        // combiner cannot signal until every core has bumped SEM_GATHER, which
        // happens after this point. That is what lets the mcast run with
        // handshake=false.
        Noc noc;
        auto sender = mcast.sender(noc);
        auto receiver = mcast.receiver(noc);

        volatile tt_l1_ptr uint32_t* gather_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));

        const uint32_t f32_tile_bytes = get_tile_size(cb_moment_total_in);
        const uint32_t moment_block_bytes = cluster_c_tiles * f32_tile_bytes;

        cb_reserve_back(cb_moment_total_in, n_total_tiles);
        const uint32_t landing_addr = get_write_ptr(cb_moment_total_in);

        if (is_combiner) {
            noc_semaphore_wait(gather_sem, hw_split_factor - 1);
            // Publish the landed remote partials to the local compute kernel.
            cb_reserve_back(cb_partial_moments, n_partial_tiles);
            cb_push_back(cb_partial_moments, n_partial_tiles);

            cb_wait_front(cb_moment_total_out, n_total_tiles);
            const uint32_t total_bytes = n_total_tiles * f32_tile_bytes;
            const uint32_t src = get_read_ptr(cb_moment_total_out);
            // Mcast1D builds the sender's rectangle by EXCLUDING the sender when
            // sender_index == 0 (mcast_host.cpp: `noc_ordered_bbox_({line_coord(1),
            // line_coord(span-1)})`), so send() never writes this core. Seed the
            // combiner's own landing slot with a local L1 copy first — measured:
            // without it cb_moment_total_in reads uninitialized L1 on the combiner
            // while cb_moment_total_out already holds the correct totals.
            noc_async_write(src, get_noc_addr(landing_addr), total_bytes);
            noc_async_write_barrier();
            sender.send(src, landing_addr, total_bytes);
            cb_pop_front(cb_moment_total_out, n_total_tiles);
        } else {
            cb_wait_front(cb_moment_sum, cluster_c_tiles);
            cb_wait_front(cb_moment_sq, cluster_c_tiles);
            const uint64_t slot = get_noc_addr(
                mcast.sender_x(),
                mcast.sender_y(),
                get_write_ptr(cb_partial_moments) + gather_slot * n_total_tiles * f32_tile_bytes);
            noc_async_write(get_read_ptr(cb_moment_sum), slot, moment_block_bytes);
            noc_async_write(get_read_ptr(cb_moment_sq), slot + moment_block_bytes, moment_block_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_moment_sum, cluster_c_tiles);
            cb_pop_front(cb_moment_sq, cluster_c_tiles);

            noc_semaphore_inc(get_noc_addr(mcast.sender_x(), mcast.sender_y(), get_semaphore(SEM_GATHER)), 1);
            noc_async_atomic_barrier();

            receiver.receive();
        }
        cb_push_back(cb_moment_total_in, n_total_tiles);
    }

    for (uint32_t u = 0; u < units; ++u) {
        const uint32_t unit = unit_start + u;
        const uint32_t n = unit / num_clusters;
        const uint32_t k = unit % num_clusters;
        const uint32_t cluster_c0_tile = k * cluster_c_tiles;

        for (uint32_t b = 0; b < num_hw_blocks; ++b) {
            const uint32_t r0 = hw_tile_start + b * BLOCK_HW_TILES;
            const uint32_t rows_this = BLOCK_HW_TILES;
            const uint32_t block_tiles = rows_this * cluster_c_tiles;

            cb_wait_front(cb_output_tiles, block_tiles);
            uint32_t l1_addr = get_read_ptr(cb_output_tiles);

            uint32_t pending = 0;
            for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
                for (uint32_t r = 0; r < rows_this; ++r) {
                    const uint32_t page = (n * tensor_hw_tiles + r0 + r) * tensor_c_tiles + cluster_c0_tile + t;
                    noc_async_write_page(page, out, l1_addr);
                    l1_addr += tile_bytes;
                    if (++pending == writes_per_barrier) {
                        noc_async_write_barrier();
                        pending = 0;
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, block_tiles);
        }
    }
}
