// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-multicaster (sender core, BRISC / NoC1; the core sits outside the worker rectangle).
// For each round r: wait for the x-reader (NCRISC / NoC0) to land round r in cb_x_resident, then
// multicast that slot to the same address on every worker (Counter data-ready, no consumer
// handshake: every slot is written once and never reused, so the source never needs protecting).
// cb_x_resident's address is identical on the sender and the workers (one CB descriptor).

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
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    const uint32_t round_bytes = round_tiles * get_tile_size(cb_x_resident);

    Noc noc;
    auto pipe = mc.sender(noc);
    for (uint32_t r = 0; r < num_k_blocks; ++r) {
        cb_wait_front(cb_x_resident, round_tiles);
        const uint32_t slot = get_read_ptr(cb_x_resident);
        pipe.template send<SourceL1Guard::CallerManaged>(slot, slot, round_bytes);
        cb_pop_front(cb_x_resident, round_tiles);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
