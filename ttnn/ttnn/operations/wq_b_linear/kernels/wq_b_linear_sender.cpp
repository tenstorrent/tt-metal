// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-sender (one core outside the worker rectangle, BRISC / NoC1).
// For each round r: read the x round (block_m_tiles x block_k_tiles tiles, (m, k) row-major)
// into slot r of cb_x_resident, then multicast the slot to the same address on every worker
// (Counter data-ready, no consumer handshake: every slot is written once and never reused).
// Round r+1's DRAM reads are issued before round r's multicast so they overlap.
// cb_x_resident is used as raw L1 here (no push/pop): its address equals the workers'.

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
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    constexpr uint32_t round_bytes = round_tiles * x_tile_bytes;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);

    Noc noc;
    auto pipe = mc.sender(noc);
    const uint32_t base = get_write_ptr(cb_x_resident);

    auto issue_round_reads = [&](uint32_t r) {
        uint32_t dst = base + r * round_bytes;
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            const uint32_t row_page = m * tensor_k_tiles + r * block_k_tiles;
            for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
                noc_async_read_tile(row_page + kk, x_acc, dst);
                dst += x_tile_bytes;
            }
        }
    };

    issue_round_reads(0);
    noc_async_read_barrier();
    for (uint32_t r = 0; r < num_k_blocks; ++r) {
        if (r + 1 < num_k_blocks) {
            issue_round_reads(r + 1);
        }
        const uint32_t slot = base + r * round_bytes;
        pipe.template send<SourceL1Guard::CallerManaged>(slot, slot, round_bytes);
        noc_async_read_barrier();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
