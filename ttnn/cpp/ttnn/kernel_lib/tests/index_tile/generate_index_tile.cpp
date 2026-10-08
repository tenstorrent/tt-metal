// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test harness for the index_tile_dataflow generator (the #58041 helper): builds every width
// tile's index tile with generate_index_tile and pushes it to CB c_0. The host pairs this with
// the house writer_unary_interleaved_start_id draining the same CB to DRAM and compares
// bit-exact against the helper's contract tile[r][c] = c + 32 * wt (tests/ttnn/unit_tests/
// kernel_lib/test_index_tile_dataflow.py).
//
// Compile-time arg 0 selects the index width in bytes (2 -> uint16_t tile, 4 -> uint32_t tile);
// runtime arg 0 is the width tile count Wt. One tile is generated per call in wt order, so the
// CB ring depth the host chooses (full or deliberately wrapped) exercises the reserve/push sync.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp"

template <typename T>
void generate_wt_tiles(uint32_t wt_dim, uint32_t start_wt) {
    for (uint32_t wt = start_wt; wt < start_wt + wt_dim; ++wt) {
        generate_index_tile<T>(0, wt);
    }
}

void kernel_main() {
    const uint32_t wt_dim = get_arg_val<uint32_t>(0);
    const uint32_t start_wt = get_arg_val<uint32_t>(1);
    constexpr uint32_t index_width_bytes = get_compile_time_arg_val(0);
    if constexpr (index_width_bytes == 4) {
        generate_wt_tiles<uint32_t>(wt_dim, start_wt);
    } else {
        generate_wt_tiles<uint16_t>(wt_dim, start_wt);
    }
}
