// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem x injector (row 0, RISCV_1 / NoC0): block op `x_round_multicast_block`.
// For every stage-1 block in send order and every K-block of it: read the round (m row tiles x s1_kb K tiles) of the
// latent from DRAM into the injector's cb_x_round slot and multicast it (Mcast helper, receivers' handshake) to the
// same slot of every stage-1 core's cb_x_round (the CB is declared on rect + injector: one address).

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
    constexpr uint32_t Q_t = get_compile_time_arg_val(2);
    constexpr uint32_t s1_kb = get_compile_time_arg_val(3);
    constexpr uint32_t nkb = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t qrt = get_compile_time_arg_val(6);
    constexpr uint32_t m = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();
    constexpr uint32_t round_tiles = m * s1_kb;
    constexpr uint32_t round_bytes = round_tiles * x_tile_bytes;
    constexpr uint32_t St = G * qrt;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(2 + i);
    }

    Noc noc;
    const auto x = TensorAccessor(x_args, x_addr, x_tile_bytes);
    auto pipe = mc.sender(noc);

    q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t, uint32_t b, uint32_t d, uint32_t r0) {
        const uint32_t row0 = b * St + d * qrt + r0;
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            cb_reserve_back(cb_x_round, round_tiles);
            const uint32_t dst = get_write_ptr(cb_x_round);
            uint32_t off = dst;
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t page0 = (row0 + i) * Q_t + kb * s1_kb;
                for (uint32_t k = 0; k < s1_kb; ++k) {
#ifndef ABLATE_X_READ
                    noc_async_read(x.get_noc_addr(page0 + k), off, x_tile_bytes);
#endif
                    off += x_tile_bytes;
                }
            }
            noc_async_read_barrier();
#ifndef ABLATE_X_MCAST
            pipe.send(dst, dst, round_bytes);  // wait every receiver's slot-free ack, multicast, signal
#endif
            cb_push_back(cb_x_round, round_tiles);
            cb_wait_front(cb_x_round, round_tiles);
            cb_pop_front(cb_x_round, round_tiles);
        }
    });
}
