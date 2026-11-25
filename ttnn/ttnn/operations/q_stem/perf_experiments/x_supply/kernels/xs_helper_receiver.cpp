// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// x_supply bench, BASELINE receiver = the op's receive_x_rounds<1> (q_stem_reader.cpp): credit slot r to the
// injector as soon as it is free (ahead = min(NI, depth) = 1), receive round r through the Mcast helper (wait Flag,
// clear it), push it to the consumer.

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
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(2);
    constexpr uint32_t ahead = get_compile_time_arg_val(3);
    constexpr uint32_t sem_x_credit = get_compile_time_arg_val(4);
    constexpr uint32_t inj_x = get_compile_time_arg_val(5);
    constexpr uint32_t inj_y = get_compile_time_arg_val(6);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast0_ct_offset"),
        get_named_compile_time_arg_val("mcast0_rt_offset")>();

    Noc noc;
    auto pipe = mc.receiver(noc);
    const uint64_t injector_credit = get_noc_addr(inj_x, inj_y, get_semaphore(sem_x_credit));
    uint32_t acked = 0;
    auto credit = [&](uint32_t r) {
        while (acked < num_rounds && acked < r + ahead) {
            const uint32_t pages = (acked - r + 1) * round_tiles;
            if (acked == r) {
                while (!cb_pages_reservable_at_back(cb_x_round, pages)) {
                }
            } else if (!cb_pages_reservable_at_back(cb_x_round, pages)) {
                break;
            }
            noc_semaphore_inc(injector_credit, 1);
            ++acked;
        }
    };
    for (uint32_t r = 0; r < num_rounds; ++r) {
        credit(r);
        pipe.receive(r);
        cb_push_back(cb_x_round, round_tiles);
    }
    noc_async_atomic_barrier();
}
