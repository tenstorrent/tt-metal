// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker writer (BRISC / NoC1).
// 1. x arrival: for each round r, wait until its sender's multicast of round r has landed in
//    cb_x_resident, then push the round's tiles to compute. With two x senders round r comes on
//    channel r % 2 ("x_mcast" / "x_mcast_b", each its own Counter: local round r / 2); with one
//    sender "x_mcast_b" is attached absent and every round comes on "x_mcast".
// 2. store: write the block_m_tiles x block_n_tiles output block to the L1-interleaved y, one final
//    barrier. out_subblock_major: compute pushes one out_subblock_h x out_subblock_w subblock at a time
//    (SubblockMajor); each is written as soon as it lands. Else one TileRowMajor block.
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
    constexpr bool out_subblock_major = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);
    constexpr bool y_store_noc_split = get_compile_time_arg_val(10) != 0;
    constexpr auto y_args = TensorAccessorArgs<11>();
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
    const uint32_t y_tile_bytes = get_tile_size(cb_out_block);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    uint32_t tile_i = 0;
    auto store_tile = [&](uint32_t m, uint32_t nn, uint32_t src) {
#ifdef ABLATE_Y_STORE
        return;
#endif
        const uint8_t noc_id = (y_store_noc_split && (tile_i++ & 1)) ? (1 - noc_index) : noc_index;
        noc_async_write(src, y_acc.get_noc_addr(m * tensor_n_tiles + cols[nn], 0, noc_id), y_tile_bytes, noc_id);
    };
    if constexpr (out_subblock_major) {
        constexpr uint32_t sb_tiles = out_subblock_h * out_subblock_w;
        const uint32_t base = get_read_ptr(cb_out_block);
        uint32_t landed = 0;
        for (uint32_t sm = 0; sm < block_m_tiles / out_subblock_h; ++sm) {
            for (uint32_t sn = 0; sn < block_n_tiles / out_subblock_w; ++sn) {
                // Cumulative wait (nothing popped until the end): subblock `landed` sits at base + landed tiles.
                uint32_t src = base + landed * y_tile_bytes;
                landed += sb_tiles;
                {
                    MaybeDeviceZoneScope("y_wait");
                    cb_wait_front(cb_out_block, landed);
                }
                MaybeDeviceZoneScope("y_issue");
                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        store_tile(sm * out_subblock_h + h, sn * out_subblock_w + w, src);
                        src += y_tile_bytes;
                    }
                }
                // The tiles stay in the CB until the barrier below (one block, never reused): pop last.
            }
        }
    } else {
        {
            MaybeDeviceZoneScope("y_wait");
            cb_wait_front(cb_out_block, out_block_tiles);
        }
        MaybeDeviceZoneScope("y_issue");
        uint32_t src = get_read_ptr(cb_out_block);
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
                store_tile(m, nn, src);
                src += y_tile_bytes;
            }
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
