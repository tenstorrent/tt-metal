// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A worker writer (RISCV_0, NoC1): my tiles - my own block's into the output, another chip's into z; then +1 on
// every carrying port's z-ready word of the row tile (this call's words: once every carrying port zeroed them).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// (fused_tail_program_factory's A2ACb: the z-ready words' CB - the same address on the ports; the zero gate's semaphore)
constexpr uint32_t CB_ZR = 5, SZR = 0;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr auto z_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(a++), z_addr = get_arg_val<uint32_t>(a++),
                   c0 = get_arg_val<uint32_t>(a++);
    const uint32_t RT = get_arg_val<uint32_t>(a++), NTH = get_arg_val<uint32_t>(a++), NTB = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++), zr_addr = get_write_ptr(CB_ZR), np = get_arg_val<uint32_t>(a++);
    const uint32_t p0 = a;
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    const auto z = TensorAccessor(z_args, z_addr, 2048);
    const uint32_t q = c0 / NTB;
    if (np > 0) {  // (every carrying port's z-ready words zeroed for this call; re-armed here)
        volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SZR));
        noc_semaphore_wait_min(g, np);
        noc_semaphore_inc(get_noc_addr(get_semaphore(SZR)), 0u - np);
        noc_async_atomic_barrier();
    }
    for (uint32_t r = 0; r < RT; ++r) {
        cb_wait_front(3, W);
        const uint32_t l1 = get_read_ptr(3);
        for (uint32_t t = 0; t < W; ++t) {
            const uint64_t dst =
                q == me ? o.get_noc_addr((me * RT + r) * NTB + c0 - q * NTB + t) : z.get_noc_addr(r * NTH + c0 + t);
            noc_async_write(l1 + t * 2048, dst, 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(3, W);
        for (uint32_t i = 0; i < np; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(p0 + i);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, zr_addr + 16 * r), 1);
        }
    }
    noc_async_atomic_barrier();
}
