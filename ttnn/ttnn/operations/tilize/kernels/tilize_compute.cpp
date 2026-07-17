// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute (TRISC unpack/math/pack).
//
// Consumes one tile-row (Wt_chunk row-major tile-pages) per tilize_block call
// from cb_rm_in and emits Wt_chunk tiled pages into cb_tiled_out. The pack-stage
// data-format reconfigure (default UnpackAndPackReconfigure) performs the
// value-preserving `dtype=` cast when cb_tiled_out's format differs from
// cb_rm_in's. fp32 uses Fp32Mode::Fast (default) — the right choice even for
// max-precision kernels (see tilize_helpers.hpp L47-71).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_tiled_out = 16;
    constexpr uint32_t Wt_chunk = get_compile_time_arg_val(0);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(1);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);  // per-core tile-row count

    compute_kernel_hw_startup(cb_rm_in, cb_tiled_out);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        compute_kernel_lib::tilize<Wt_chunk, cb_rm_in, cb_tiled_out>(num_blocks);
    }
}
