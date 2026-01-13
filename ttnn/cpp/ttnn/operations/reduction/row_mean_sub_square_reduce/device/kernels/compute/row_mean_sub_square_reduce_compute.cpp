// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"

// STUB COMPUTE: Simple passthrough from input CB to output CB
// This stub just copies data through without computation
// Real implementation (Stage 7) will do: tilize → reduce → sub → square → reduce → untilize
//
// For the stub: input is N*C*H sticks of width W, output is N*C*H sticks of width 32
// Each tile-row has 32 sticks. We just copy data through.

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t num_output_sticks = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;    // rm_in
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // rm_out

    // Passthrough: For each output stick, consume one input stick and copy
    copy_tile_init(cb_in);

    for (uint32_t i = 0; i < num_output_sticks; i++) {
        acquire_dst();
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // Copy from input CB to DST register, then pack to output CB
        copy_tile(cb_in, 0, 0);  // Copy from CB to DST[0]
        pack_tile(0, cb_out);    // Pack DST[0] to output CB

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        release_dst();
    }
}

}  // namespace NAMESPACE
