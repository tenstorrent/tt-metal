// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// riscv_transpose microbench, BRISC / NCRISC: `reps` transposes of one unit held in this RISC's own
// region of CB 0 (no NoC traffic), one zone per unit. Values are whatever L1 holds: timing only.
#include "api/dataflow/dataflow_api.h"
#include "micro_body.hpp"

constexpr uint32_t region = get_compile_time_arg_val(0);
constexpr uint32_t region_bytes = get_compile_time_arg_val(1);
constexpr uint32_t reps = get_compile_time_arg_val(2);
constexpr uint32_t pages = get_compile_time_arg_val(3);
constexpr uint32_t values = get_compile_time_arg_val(4);
constexpr uint32_t src_stride = get_compile_time_arg_val(5);
constexpr uint32_t dst_stride = get_compile_time_arg_val(6);
constexpr uint32_t impl = get_compile_time_arg_val(7);
constexpr uint32_t batch = get_compile_time_arg_val(8);

void kernel_main() {
    const uint32_t src = get_write_ptr(0) + region * region_bytes;
    const uint32_t dst = src + pages * src_stride;
    for (uint32_t i = 0; i < reps; ++i) {
        DeviceZoneScopedN("RVT-UNIT");
        asm volatile("" ::: "memory");
        const uint32_t acc = rvt_micro::run_unit<pages, values, src_stride, dst_stride, impl, batch>(src, dst);
        asm volatile("" ::"r"(acc));
        asm volatile("" ::: "memory");
    }
}
