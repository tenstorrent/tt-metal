// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// TILIZE_HANG_DEBUG: temporary instrumentation to localize the tilize compute hang.
// Must be defined BEFORE including tilize_helpers so the per-phase markers inside the
// helper are compiled in. Requires the DPRINT server to be enabled at runtime, e.g.
//   export TT_METAL_DPRINT_CORES=all
// Remove this define (and the markers below) once the hang is understood.
#define TILIZE_HANG_DEBUG 1

#include "api/debug/dprint.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    // Whether this configuration will take the fast-tilize path (mirrors the predicate the
    // helper uses internally). This is what routes a 1-tile-wide (full_dim==1) bf16 tilize
    // into the fragile WH fast-tilize path that Blackhole explicitly avoids.
    constexpr bool dbg_use_fast =
        compute_kernel_lib::can_use_fast_tilize<per_core_block_tile_cnt, tt::CBIndex::c_0, tt::CBIndex::c_16>();

    DPRINT_UNPACK(
        "TZ U: enter block_cnt={} full_dim={} use_fast={}\n",
        per_core_block_cnt,
        per_core_block_tile_cnt,
        (uint32_t)dbg_use_fast);
    DPRINT_MATH(
        "TZ M: enter block_cnt={} full_dim={} use_fast={}\n",
        per_core_block_cnt,
        per_core_block_tile_cnt,
        (uint32_t)dbg_use_fast);
    DPRINT_PACK(
        "TZ P: enter block_cnt={} full_dim={} use_fast={}\n",
        per_core_block_cnt,
        per_core_block_tile_cnt,
        (uint32_t)dbg_use_fast);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    DPRINT_UNPACK("TZ U: hw_startup done\n");
    DPRINT_MATH("TZ M: hw_startup done\n");
    DPRINT_PACK("TZ P: hw_startup done\n");

    // Use lossless tilize for fp32 inputs to preserve exact values (fast tilize truncates fp32 → tf32)
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<tt::CBIndex::c_0>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        tt::CBIndex::c_0,
        tt::CBIndex::c_16,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);

    DPRINT_UNPACK("TZ U: tilize() returned\n");
    DPRINT_MATH("TZ M: tilize() returned\n");
    DPRINT_PACK("TZ P: tilize() returned\n");
}
