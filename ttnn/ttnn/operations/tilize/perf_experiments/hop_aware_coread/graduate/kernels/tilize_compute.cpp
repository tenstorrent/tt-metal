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
//
// Numeric formats (CT `fp32_lossless`): cb_input_sticks carries the input dtype and
// cb_output_tiles the output dtype; the value-preserving cast happens at pack. A 32-bit
// input (Float32 / Int32 / UInt32) is tagged UnpackToDestFp32 on the host and runs with
// fp32_dest_acc_en=true, so the helper is asked for Fp32Mode::Lossless: tilize IS the final
// consumer here, so the fast path's fp32 -> tf32 truncation would corrupt the output.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

using namespace compute_kernel_lib::tilize_config;

template <uint32_t block_width, uint32_t cb_in, uint32_t cb_out, Fp32Mode fp32_mode>
FORCE_INLINE void tilize_one_row(bool first, bool last) {
    constexpr auto reconfig = ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure;
    if (first && last) {
        compute_kernel_lib::
            tilize<block_width, cb_in, cb_out, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, reconfig, fp32_mode>(
                1);
    } else if (first) {
        compute_kernel_lib::
            tilize<block_width, cb_in, cb_out, InitUninitMode::InitOnly, WaitMode::WaitBlock, reconfig, fp32_mode>(1);
    } else if (last) {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            InitUninitMode::UninitOnly,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            fp32_mode>(1);
    } else {
        compute_kernel_lib::tilize<
            block_width,
            cb_in,
            cb_out,
            InitUninitMode::Neither,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            fp32_mode>(1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width = get_compile_time_arg_val(2);
    constexpr bool split_reader = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_input_sticks_odd = get_compile_time_arg_val(4);
    constexpr Fp32Mode fp32_mode = get_compile_time_arg_val(5) != 0 ? Fp32Mode::Lossless : Fp32Mode::Fast;

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(1);

    const uint32_t num_col_blocks = (core_col_tiles + block_width - 1) / block_width;
    const uint32_t num_blocks = core_row_tiles * num_col_blocks;

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);
    // OCCUPANCY, not payload: the helper does its own per-block cb_wait_front (unpack) and
    // cb_reserve_back (pack), so this zone includes starvation on the reader and back-pressure
    // from the writer.
    MaybeDeviceZoneScope("compute_tilize");
    if constexpr (!split_reader) {
        compute_kernel_lib::tilize<
            block_width,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            fp32_mode>(num_blocks);
    } else {
        for (uint32_t seq = 0; seq < num_blocks; ++seq) {
            const bool first = seq == 0;
            const bool last = seq + 1 == num_blocks;
            if (seq & 1) {
                tilize_one_row<block_width, cb_input_sticks_odd, cb_output_tiles, fp32_mode>(first, last);
            } else {
                tilize_one_row<block_width, cb_input_sticks, cb_output_tiles, fp32_mode>(first, last);
            }
        }
    }
}
