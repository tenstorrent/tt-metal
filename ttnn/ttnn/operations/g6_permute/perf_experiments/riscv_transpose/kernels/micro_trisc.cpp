// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// riscv_transpose microbench, TRISC 0/1/2: as micro_dm.cpp, region 2 + TRISC index, on the TRISCs
// selected by the mask (bit t = TRISC t). Plain RISC-V loads and stores only: no unpacker, FPU or packer.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "micro_body.hpp"

constexpr uint32_t trisc_mask = get_compile_time_arg_val(0);
constexpr uint32_t region_bytes = get_compile_time_arg_val(1);
constexpr uint32_t reps = get_compile_time_arg_val(2);
constexpr uint32_t pages = get_compile_time_arg_val(3);
constexpr uint32_t values = get_compile_time_arg_val(4);
constexpr uint32_t src_stride = get_compile_time_arg_val(5);
constexpr uint32_t dst_stride = get_compile_time_arg_val(6);
constexpr uint32_t impl = get_compile_time_arg_val(7);
constexpr uint32_t batch = get_compile_time_arg_val(8);

#if defined(TRISC_UNPACK)
constexpr uint32_t trisc = 0;
#elif defined(TRISC_MATH)
constexpr uint32_t trisc = 1;
#else
constexpr uint32_t trisc = 2;
#endif

void kernel_main() {
    const uint32_t base = ckernel::get_tile_address(0, 0);
    if constexpr ((trisc_mask >> trisc) & 1) {
        const uint32_t src = base + (2 + trisc) * region_bytes;
        const uint32_t dst = src + pages * src_stride;
        for (uint32_t i = 0; i < reps; ++i) {
            DeviceZoneScopedN("RVT-UNIT");
            asm volatile("" ::: "memory");
            const uint32_t acc = rvt_micro::run_unit<pages, values, src_stride, dst_stride, impl, batch>(src, dst);
            asm volatile("" ::"r"(acc));
            asm volatile("" ::: "memory");
        }
    }
}
