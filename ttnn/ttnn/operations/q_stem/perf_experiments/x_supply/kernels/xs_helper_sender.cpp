// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x_supply bench, BASELINE injector multicast half = the op's q_stem_injector.cpp (X_INJECTORS = 1):
// kernel_lib Mcast helper (fixed sender, handshake off, Flag data-ready, linked), flow control by the receivers'
// cumulative slot credits on sem_x_credit (one round credited ahead: the Flag must be seen and cleared before the
// next round is sent).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(3);
    constexpr uint32_t depth = get_compile_time_arg_val(4);
    constexpr uint32_t receivers = get_compile_time_arg_val(5);
    constexpr uint32_t sem_x_credit = get_compile_time_arg_val(6);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast0_ct_offset"),
        get_named_compile_time_arg_val("mcast0_rt_offset")>();
    constexpr uint32_t round_bytes = round_tiles * x_tile_bytes;

    Noc noc;
    auto pipe = mc.sender(noc);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x_credit));
    uint32_t base = 0;
    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_wait_front(cb_x_round, round_tiles);
        const uint32_t src = get_read_ptr(cb_x_round);
        if (r == 0) {
            base = src;
        }
        noc_semaphore_wait_min(credit, receivers * (r + 1));
        pipe.send(src, base + (r % depth) * round_bytes, round_bytes);
        cb_pop_front(cb_x_round, round_tiles);
    }
}
