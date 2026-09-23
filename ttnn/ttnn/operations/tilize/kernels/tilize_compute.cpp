// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute (TRISC0/1/2) — the `tilize_block` block operation (op_design.md).
//
// Default: ONE compute_kernel_lib::tilize call covers every block of this Tensix
// core, so tilize init / reconfig / uninit happen once per kernel. The helper
// processes core_row_tiles * num_col_blocks helper-blocks of block_width tiles;
// every quantum is the nominal block_width (the ragged last column block
// tilizes stale tail columns that the writer never writes).
//
// Split reader (CT `split_reader`): the input arrives through two CBs (even
// sequence positions from NCRISC, odd from BRISC — one producer each), which a
// single helper call cannot alternate between. The same helper is called once
// per tile-row with its documented back-to-back lifecycle: init (and the
// unpack/pack reconfig) on the first call only, InitUninitMode::Neither +
// NoReconfigure in the middle, uninit on the last. Both input CBs have identical
// format and tile geometry, so the one init configures both.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace {

using namespace compute_kernel_lib::tilize_config;

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out>
FORCE_INLINE void tilize_one_row(bool first, bool last) {
    if (first && last) {
        compute_kernel_lib::tilize<block_width, cb_in, cb_out, InitUninitMode::InitAndUninit>(1);
    } else if (first) {
        compute_kernel_lib::tilize<block_width, cb_in, cb_out, InitUninitMode::InitOnly>(1);
    } else if (last) {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            InitUninitMode::UninitOnly,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    } else {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            InitUninitMode::Neither,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width = get_compile_time_arg_val(2);
    constexpr bool split_reader = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_input_sticks_odd = get_compile_time_arg_val(4);

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(1);

    const uint32_t num_col_blocks = (core_col_tiles + block_width - 1) / block_width;
    const uint32_t num_blocks = core_row_tiles * num_col_blocks;

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);
    if constexpr (!split_reader) {
        compute_kernel_lib::tilize<block_width, cb_input_sticks, cb_output_tiles>(num_blocks);
    } else {
        for (uint32_t seq = 0; seq < num_blocks; ++seq) {
            const bool first = seq == 0;
            const bool last = seq + 1 == num_blocks;
            if (seq & 1) {
                tilize_one_row<block_width, cb_input_sticks_odd, cb_output_tiles>(first, last);
            } else {
                tilize_one_row<block_width, cb_input_sticks, cb_output_tiles>(first, last);
            }
        }
    }
}
