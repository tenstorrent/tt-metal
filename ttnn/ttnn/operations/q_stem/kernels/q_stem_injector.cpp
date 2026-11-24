// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem x injector, multicast half (row 0, RISCV_1): the send side of `x_round_multicast_block`.
// X_INJECTORS rotating injectors (each its own Mcast family, alternating NoCs): injector `idx` owns the global
// rounds r = idx, idx + NI, ... Every round its DRAM half (q_stem_injector_reader.cpp) pushed into cb_x_round is
// multicast (Mcast helper, Counter data-ready signal) into slot r % x_round_depth of every stage-1 core's
// cb_x_round (the CB is declared on rect + injectors: one base address). Flow control: the receivers credit each
// slot to its injector as soon as it is free (cumulative count on this core's credit semaphore), so the injectors'
// rounds overlap instead of waiting for a per-round handshake.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/ttnn/operations/q_stem/kernels/q_stem_common.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(3);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val(Q_STEM_X_FAMILY_ARG("_ct_offset")),
        get_named_compile_time_arg_val(Q_STEM_X_FAMILY_ARG("_rt_offset"))>();
    constexpr uint32_t idx = get_compile_time_arg_val(4);  // this injector
    constexpr uint32_t num_injectors = get_compile_time_arg_val(5);
    constexpr uint32_t depth = get_compile_time_arg_val(6);      // x_round_depth (receivers' slots)
    constexpr uint32_t receivers = get_compile_time_arg_val(7);  // credits per round
    constexpr uint32_t sem_x_credit = get_compile_time_arg_val(8);
    constexpr uint32_t round_bytes = round_tiles * x_tile_bytes;

    Noc noc;
    auto pipe = mc.sender(noc);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x_credit));
    uint32_t base = 0;  // receivers' slot 0 (= this core's cb_x_round base: the CB is fresh at the first wait)
    uint32_t k = 0;
    for (uint32_t r = idx; r < num_rounds; r += num_injectors, ++k) {
        cb_wait_front(cb_x_round, round_tiles);
        const uint32_t src = get_read_ptr(cb_x_round);
        if (k == 0) {
            base = src;
        }
        noc_semaphore_wait_min(credit, receivers * (k + 1));  // every receiver's slot r % depth is free
#ifndef ABLATE_X_MCAST
        pipe.send(src, base + (r % depth) * round_bytes, round_bytes);  // multicast, then data-ready counter
#endif
        cb_pop_front(cb_x_round, round_tiles);
    }
}
