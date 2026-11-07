// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A planner-only core's kernel (token loops: the routing planner's instances past the router cores, BRISC and NCRISC):
// once every router core's rows are written (RA_NP_ increments of my rows word), ra::plan. Args: my rows word, my
// instance's region (0: none here), then ra::plan's.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/router_addr.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_plan.hpp"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0), B = get_arg_val<uint32_t>(1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), RA_NP_);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows) = 0;  // (re-arm: all increments landed)
    if (B) {
        DeviceZoneScopedN("SE-RPLAN");
        ra::plan(2, B);
    }
}
