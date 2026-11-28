// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf-1 experiment tensix_untilize_bf16: untilize compute of the bf16 packer path.
//
// cb_tile_window (tile pages) -> unpack A -> MOVA2D -> pack-untilize -> cb_row_block (tile pages laid
// out as 32 rows of block_cols_tiles * 64 B per tile-row). Same structure as R3's
// g6_to_layout_untilize_compute.cpp (compute_kernel_lib::untilize, init/uninit once per kernel).
//
// Bit identity comes from the HOST side, not from this kernel: both CBs declare DataFormat::UInt16 for
// the bf16 bytes. The unpacker, SrcA ("Integer 16"), MOVA2D and the packer then move integers, and
// hw_configure already selects the preserve Src zero flag for a UInt16 operand
// (ckernel::requires_disabled_src_zero_flag), so all 65536 bf16 patterns survive. With Float16_b CBs
// the float conversion of the unpack/pack path turns -0.0 and subnormals into +0 and NaN into +-inf
// whatever the zero flag (509 of 65536 patterns), which is what the optional pin below was built to
// test (CT arg preserve_zero_flag = 1: one raw MATH call after untilize_init; it measurably changes
// nothing and is off in the recommended variant, which is helper-only).

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;
namespace ucfg = compute_kernel_lib::untilize_config;

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
    constexpr uint32_t preserve_zero_flag = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr uint32_t block_cols_tiles = Geometry::block_cols_tiles;

    compute_kernel_hw_startup(cb_tile_window, cb_row_block);
    compute_kernel_lib::untilize_init<block_cols_tiles, cb_tile_window, cb_row_block>();
    if constexpr (preserve_zero_flag != 0) {
        MATH((ckernel::math::_configure_preserve_zero_flag_state_()));
    }

    CoreBlocks<false> blocks;
    Block block;
    uint32_t tile_rows = 0;
    while (blocks.next(block)) {
        tile_rows += Geometry::block_rows_tiles;
    }
    if (tile_rows > 0) {
        compute_kernel_lib::untilize<
            block_cols_tiles,
            cb_tile_window,
            cb_row_block,
            ucfg::InitUninitMode::Neither,
            ucfg::WaitMode::WaitBlock,
            ucfg::ReconfigureRegisterDatatypeMode::NoReconfigure>(tile_rows);
    }

    compute_kernel_lib::untilize_uninit<block_cols_tiles, cb_tile_window, cb_row_block>();
}
