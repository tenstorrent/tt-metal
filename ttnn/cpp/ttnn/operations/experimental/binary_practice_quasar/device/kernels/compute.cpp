// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute: out = a + b, one tile at a time, on the FPU.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr auto a_id = static_cast<uint32_t>(dfb::a);
    constexpr auto b_id = static_cast<uint32_t>(dfb::b);
    constexpr auto out_id = static_cast<uint32_t>(dfb::out);

    compute_kernel_hw_startup(a_id, b_id, out_id);
    add_init(a_id, b_id);

    DataflowBuffer dfb_a(a_id);
    DataflowBuffer dfb_b(b_id);
    DataflowBuffer dfb_out(out_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_a.wait_front(1);
        dfb_b.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        add_tiles(a_id, b_id, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, out_id);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_a.pop_front(1);
        dfb_b.pop_front(1);
    }
}
