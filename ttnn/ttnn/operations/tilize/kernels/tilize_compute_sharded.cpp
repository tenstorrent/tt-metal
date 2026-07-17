// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute (sharded, zero-copy) — TRISC unpack/math/pack.
//
// Same-spec sharded I/O path: both cb_rm_in and cb_tiled_out are aliased
// directly onto the local L1 shard buffers (cb_descriptor_from_sharded_tensor),
// so there is NO reader and NO writer kernel — no NoC transfers at all.
//
// The ROW_MAJOR input shard is a contiguous `shard_h x shard_w` block; with the
// input CB's page_size overridden to tile_size, each group of Wt pages is
// exactly one tile-row (32 contiguous rows of Wt*32 elements) — precisely what
// tilize_block consumes. tilize_block packs the Wt output tiles straight into
// the output shard's L1 (row-major tile order), reconstructing the same logical
// region in TILE layout. Because input and output use the IDENTICAL shard spec,
// each core independently tilizes its own block and the global layout is
// preserved regardless of sharding scheme / orientation.
//
// The input shard data is already resident in L1 (nobody pushes it), so the
// compute kernel arms the aliased input CB with a single self reserve+push of
// the whole shard before handing it to the tilize helper (the resident-sharded
// pattern used by examples/reduce_block).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_tiled_out = 16;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);          // tiles per tile-row (shard_w / 32)
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);  // tile-rows in the shard (shard_h / 32)
    constexpr uint32_t is_fp32_in = get_compile_time_arg_val(2);
    constexpr uint32_t in_pages = num_blocks * Wt;

    compute_kernel_hw_startup(cb_rm_in, cb_tiled_out);

    // Arm the resident RM input shard (aliased CB — no reader pushes it).
    cb_reserve_back(cb_rm_in, in_pages);
    cb_push_back(cb_rm_in, in_pages);

    if constexpr (is_fp32_in) {
        // Bit-exact fp32 tilize (terminal op): Fp32Mode::Lossless + UnpackToDestFp32
        // (set on cb_rm_in in the descriptor) + fp32_dest_acc_en.
        compute_kernel_lib::tilize<
            Wt,
            cb_rm_in,
            cb_tiled_out,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            compute_kernel_lib::tilize_config::Fp32Mode::Lossless>(num_blocks);
    } else {
        // bf16 / integer passthrough: default Fast path; pack reconfigure drives
        // any value-preserving dtype= cast.
        compute_kernel_lib::tilize<Wt, cb_rm_in, cb_tiled_out>(num_blocks);
    }
}
