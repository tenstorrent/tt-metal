// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker writer (BRISC / NoC1).
// 1. x arrival: for each round r, wait until the sender's multicast of round r has landed in
//    cb_x_resident (Counter >= r+1), then push the round's tiles to compute.
// 2. store: wait for the block_m_tiles x block_n_tiles output block (TileRowMajor) and write it
//    to the L1-interleaved y, one barrier.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
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
    constexpr auto y_args = TensorAccessorArgs<7>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

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
    for (uint32_t r = 0; r < num_k_blocks; ++r) {
        cb_reserve_back(cb_x_resident, round_tiles);  // never blocks: x is fully resident
        pipe.receive(r);
        cb_push_back(cb_x_resident, round_tiles);
    }

    // ---- store_block ----
    const uint32_t y_tile_bytes = get_tile_size(cb_out_block);
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    cb_wait_front(cb_out_block, out_block_tiles);
    uint32_t src = get_read_ptr(cb_out_block);
    for (uint32_t m = 0; m < block_m_tiles; ++m) {
        for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
            noc_async_write_tile(m * tensor_n_tiles + cols[nn], y_acc, src);
            src += y_tile_bytes;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_block, out_block_tiles);
}
