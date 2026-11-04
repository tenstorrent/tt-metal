// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather's reader on the port cores (xg_ct.hpp, xg_walk.hpp, xg_la_reader.hpp, xg_rs_reader.hpp).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_ct.hpp"
constexpr auto x_args = TensorAccessorArgs<14>();
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_walk.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_la_reader.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/xg_rs_reader.hpp"

void kernel_main() {
    size_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++), xown = get_arg_val<uint32_t>(a++),
                   v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++),
                   up_ipc = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, T = ent + 4 * n;  // entries (q, part, up_base within a chunk, packets a chunk); tables
    // (after the tables: the engine's "own shard loaded" counter here and its full count)
    const uint32_t xl0 = T + G + S0 + S1 + 3 * GX + GY;
    volatile tt_l1_ptr uint32_t* xloaded = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(xl0));
    const uint32_t xl_need = get_arg_val<uint32_t>(xl0 + 1);
    bool xl = false;
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    uint32_t batch = 0, wptr = 0;
    for (uint32_t m = 0; m < NCH; ++m) {
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = get_arg_val<uint32_t>(ent + 4 * k), part = get_arg_val<uint32_t>(ent + 4 * k + 1);
            const uint32_t up_base = get_arg_val<uint32_t>(ent + 4 * k + 2),
                           npk = get_arg_val<uint32_t>(ent + 4 * k + 3);
            uint32_t done = 0;
            walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t kj, uint32_t kk0, uint32_t np, uint32_t idx) {
                if (batch == 0) {
                    cb_reserve_back(0, RP * GROUP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t dst = wptr + batch * RP * 2048;
                if (k == 0) {  // my own shard: from the engine's x-own L1 once every engine core loaded it (no DRAM
                               // here: the DRAM is the engine's weight load's)
                    if (!xl) {
                        noc_semaphore_wait_min(xloaded, xl_need);
                        xl = true;
                    }
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * 2048);
                } else {  // relay: upstream's packet idx of (chunk m, its entry k - 1) landed here
                    noc_semaphore_wait_min(arrived, m * up_ipc + up_base + idx / INC + 1);
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * 2048);
                }
                ++batch;
                ++done;
                if (batch == GROUP || done == npk) {
                    noc_async_read_barrier();
                    cb_push_back(0, RP * GROUP);
                    batch = 0;
                }
            });
        }
    }
    noc_semaphore_wait_min(xloaded, xl_need);  // (re-arm: every engine core counted itself on every port)
    noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(xloaded)), 0u - xl_need);
    noc_async_atomic_barrier();
    const uint32_t rsa = xl0 + 2, rnu = get_arg_val<uint32_t>(rsa), rnc = get_arg_val<uint32_t>(rsa + 1 + 2 * rnu);
    uint32_t pos = 0;  // (the gather left the CB batch-aligned; the phases after it share one chunk count)
    uint32_t seen[128], lseen[128];
    for (uint32_t m = 0; m < NCH; ++m) {
        la_gather(rsa + 2 + 2 * rnu + 2 * rnc + G + RSX, pos, m, lseen);
        rs_gather(rsa, pos, m, seen);
    }
}
