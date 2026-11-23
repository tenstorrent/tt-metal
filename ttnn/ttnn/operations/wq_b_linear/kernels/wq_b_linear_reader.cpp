// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker reader (NCRISC / NoC0): stream this worker's block_n_tiles W columns,
// one K-step (block_k_tiles x block_n_tiles tiles, (k, n_local) row-major) per barrier/push.
// w_wait_x0: hold the W stream until x round 0 has landed here (x channel "x_mcast", Counter >= 1),
// so the x senders' first DRAM reads do not share the banks with all 64 W streams; compute cannot
// start before x round 0 anyway.

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
    constexpr uint32_t cb_w_block = get_compile_time_arg_val(0);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(4);
    constexpr bool w_wait_x0 = get_compile_time_arg_val(5) != 0;
    constexpr auto w_args = TensorAccessorArgs<6>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

    constexpr uint32_t w_block_tiles = block_k_tiles * block_n_tiles;

    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    uint32_t cols[block_n_tiles];
    for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
        cols[nn] = get_arg_val<uint32_t>(1 + nn);
    }

    const uint32_t w_tile_bytes = get_tile_size(cb_w_block);
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);

    if constexpr (w_wait_x0) {
        Noc noc;
        mc.receiver(noc).receive(0);  // Counter wait only; the writer's own receive(0) is unaffected
    }
    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        cb_reserve_back(cb_w_block, w_block_tiles);
        uint32_t dst = get_write_ptr(cb_w_block);
        for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
            const uint32_t row_page = (kb * block_k_tiles + kk) * tensor_n_tiles;
            for (uint32_t nn = 0; nn < block_n_tiles; ++nn) {
                noc_async_read_tile(row_page + cols[nn], w_acc, dst);
                dst += w_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w_block, w_block_tiles);
    }
}
