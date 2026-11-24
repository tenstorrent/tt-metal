// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem x injector, multicast half (row 0, RISCV_1 / NoC0): the send side of `x_round_multicast_block`.
// Every round the DRAM half (q_stem_injector_reader.cpp) pushed into cb_x_round is multicast (Mcast helper, receivers'
// handshake) to the same slot of every stage-1 core's cb_x_round (the CB is declared on rect + injector: one address,
// and the injector's slot sequence is the receivers').

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
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();
    constexpr uint32_t round_bytes = round_tiles * x_tile_bytes;

    Noc noc;
    auto pipe = mc.sender(noc);
    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_wait_front(cb_x_round, round_tiles);
        const uint32_t src = get_read_ptr(cb_x_round);
#ifndef ABLATE_X_MCAST
        pipe.send(src, src, round_bytes);  // wait every receiver's slot-free ack, multicast, signal
#endif
        cb_pop_front(cb_x_round, round_tiles);
    }
}
