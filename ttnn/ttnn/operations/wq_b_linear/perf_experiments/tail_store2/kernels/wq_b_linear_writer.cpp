// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker writer (BRISC / NoC1).
// 1. x arrival: for each round r, wait until its sender's multicast of round r has landed in
//    cb_x_resident, then push the round's tiles to compute. With two x senders round r comes on
//    channel r % 2 ("x_mcast" / "x_mcast_b", each its own Counter: local round r / 2); with one
//    sender "x_mcast_b" is attached absent and every round comes on "x_mcast".
// 2. store: write the block_m_tiles x block_n_tiles output block (TileRowMajor, one push from compute)
//    to the L1-interleaved y, one final barrier.
//    y_store_noc_split: every other tile goes out on NoC0 (idle by then: the W reads are done), the
//    rest on this kernel's NoC1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
// Zones (permanent): x_wait_r0 (x round 0 arrival), x_wait_rest (rounds 1..), y_wait (waiting on compute's
// output; cumulative over subblocks), y_issue, y_barrier.
// Ablation: ABLATE_Y_STORE skips the y writes (waits, barrier and pop kept).
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_block = get_compile_time_arg_val(1);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(6);
    constexpr bool y_store_noc_split = get_compile_time_arg_val(7) != 0;
    constexpr auto y_args = TensorAccessorArgs<8>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();
    constexpr auto mc_b = McastArgs<
        get_named_compile_time_arg_val("x_mcast_b_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_b_rt_offset")>();
    constexpr uint32_t num_x_senders = mc_b.active ? 2 : 1;

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    constexpr uint32_t out_block_tiles = block_m_tiles * block_n_tiles;

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    uint32_t cols[block_n_tiles];
    for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
        cols[nn] = get_arg_val<uint32_t>(1 + nn);
    }

    // ---- tail_store2: output tile order / store groups ----
    // Group = tiles compute pushes at once: whole block (op), 1 or 2 (TAIL_LAST), one column (TAIL_RAWCOL).
#if defined(TAIL_RAWCOL)
    constexpr bool col_major = true;
    constexpr uint32_t group = block_m_tiles;
#elif defined(TAIL_LAST)
    constexpr bool col_major = false;
    constexpr uint32_t group = TAIL_LAST;
#else
    constexpr bool col_major = false;
    constexpr uint32_t group = out_block_tiles;
#endif
    const uint32_t y_tile_bytes = get_tile_size(cb_out_block);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    auto tile_page = [&](uint32_t i) {
        const uint32_t m = col_major ? i % block_m_tiles : i / block_n_tiles;
        const uint32_t nn = col_major ? i / block_m_tiles : i % block_n_tiles;
        return m * tensor_n_tiles + cols[nn];
    };
    auto tile_noc = [&](uint32_t i) -> uint8_t { return (y_store_noc_split && (i & 1)) ? (1 - noc_index) : noc_index; };
#ifdef TAIL_PRECOMP
    // Resolve every output tile's NoC address (TensorAccessor bank lookup) before the x waits, while BRISC idles.
    uint64_t dst_addr[out_block_tiles];
    for (uint32_t i = 0; i < out_block_tiles; ++i) {
        dst_addr[i] = y_acc.get_noc_addr(tile_page(i), 0, tile_noc(i));
    }
#define TAIL_DST(i) dst_addr[i]
#else
#define TAIL_DST(i) y_acc.get_noc_addr(tile_page(i), 0, tile_noc(i))
#endif

    // ---- receive_x_rounds ----
    Noc noc;
    auto pipe = mc.receiver(noc);
    auto pipe_b = mc_b.optional_receiver(noc);
    auto receive_round = [&](uint32_t r) {
        cb_reserve_back(cb_x_resident, round_tiles);  // never blocks: x is fully resident
        if (num_x_senders == 1 || (r % num_x_senders) == 0) {
            pipe.receive(r / num_x_senders);
        } else {
            pipe_b->receive(r / num_x_senders);
        }
        cb_push_back(cb_x_resident, round_tiles);
    };
    {
        MaybeDeviceZoneScope("x_wait_r0");
        receive_round(0);
    }
    {
        MaybeDeviceZoneScope("x_wait_rest");
        for (uint32_t r = 1; r < num_k_blocks; ++r) {
            receive_round(r);
        }
    }

    // ---- store_block ----
    const uint32_t base = get_read_ptr(cb_out_block);
    auto issue = [&](uint32_t t0) {
#ifndef ABLATE_Y_STORE
        for (uint32_t t = t0; t < t0 + group; ++t) {
            noc_async_write(base + t * y_tile_bytes, TAIL_DST(t), y_tile_bytes, tile_noc(t));
        }
#endif
    };
    {
        MaybeDeviceZoneScope("y_wait");  // first group only (per-group zones would sit on the critical path)
        cb_wait_front(cb_out_block, group);
    }
    {
        MaybeDeviceZoneScope("y_issue");  // issue of group 0, then wait + issue of the rest
        issue(0);
        for (uint32_t t0 = group; t0 < out_block_tiles; t0 += group) {
            cb_wait_front(cb_out_block, t0 + group);
            issue(t0);
        }
    }
    {
        MaybeDeviceZoneScope("y_barrier");
        noc_async_write_barrier();
        if constexpr (y_store_noc_split) {
            noc_async_write_barrier(1 - noc_index);
        }
    }
    cb_pop_front(cb_out_block, out_block_tiles);
}
