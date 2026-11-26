// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu compute.
//
// Per block (op_design.md -> Block schedule):
//   untilize_rows_block   (TILE regime) compute_kernel_lib::untilize: input tiles -> row-major
//                         tile-rows (the reader copies them into the cb_rows slot).
//   windowed_tilize_block compute_kernel_lib::tilize with stick pages: per tile-row 4 windows,
//                         window p = sticks [32m + p, 32m + p + 32) = x[t - 3 + p]. The shift is a
//                         slide-by-pop on the slot (pops 1,1,1,29; 1,1,1,32 on the block's last
//                         tile-row) — exact data movement, fidelity-independent.
//   conv_silu_block       y = silu(sum_p window_p * rowbcast(tap tile, row 2p)) in DEST, one pack;
//                         the MACs run on the math thread, SiLU on the packer thread.
//
// RAW-LLK DEVIATION (conv_silu_block): eltwise_chain cannot express it.
//   (1) chain.inl:3397 — in a DEST-accumulating chain every DEST-only (SFPU) element runs on a
//       transient lane at slot >= 1, never on the sticky accumulated D0, so "accumulate 4 products,
//       then SiLU the sum" is not one chain;
//   (2) BinaryFpu::exec calls mul_tiles_bcast<bt>(...) without bcast_row_idx (chain.inl:1469), so the
//       packed tap tile (tap p in row 2p) is not addressable.
//   Two chains would need a pack + unpack of the pre-activation through an extra CB (rejected:
//   fold into the accumulator). Gap closer: a chain option to apply post-accumulation DEST-only
//   elements on the sticky slot, plus bcast_row_idx on BinaryFpu.
//   The block op below relies on ELWMUL accumulating into DEST (llk_math_eltwise_binary.h: the
//   standard ELWMUL path issues no ZEROACC; fidelity phases accumulate) and on DEST being zero at
//   acquire (cleared by the packer on release) — the same precondition every mul_tiles has.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {

constexpr uint32_t cb_rows = 0;
constexpr uint32_t cb_tap_tiles = 1;
constexpr uint32_t cb_window_tiles = 2;
constexpr uint32_t cb_output_tiles = 3;
constexpr uint32_t cb_input_tiles = 4;
constexpr uint32_t cb_untilized_rows = 5;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t TAP_COUNT = 4;
constexpr uint32_t HALO_ROWS = TAP_COUNT - 1;
constexpr uint32_t WINDOW_STICKS = TILE_DIM + HALO_ROWS;  // every window of a tile-row lies in these
constexpr uint32_t TAP_ROW_STRIDE = 2;                    // tap p lives in row 2p of the packed tap tile

using compute_kernel_lib::tilize_config::InitUninitMode;
using compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode;
using compute_kernel_lib::tilize_config::WaitMode;

// One window: tilize 32 sticks from the slot's read pointer, then pop `pages` sticks.
template <uint32_t block_ch_tiles, InitUninitMode init_mode, ReconfigureRegisterDatatypeMode reconfig>
ALWI void tilize_window(DataflowBuffer& rows, DataflowBuffer& windows, uint32_t pages) {
    compute_kernel_lib::tilize<block_ch_tiles, cb_rows, cb_window_tiles, init_mode, WaitMode::NoWait, reconfig>(
        rows, windows, 1, pages);
}

// windowed_tilize_block: rows_this_block tile-rows x 4 windows x block_ch_tiles tiles, layout
// [m][p][c] in cb_window_tiles. One tilize init / uninit per block; one wait per tile-row.
template <uint32_t block_ch_tiles>
void windowed_tilize_block(DataflowBuffer& rows, DataflowBuffer& windows, uint32_t rows_this_block) {
    constexpr auto Keep = ReconfigureRegisterDatatypeMode::NoReconfigure;
    for (uint32_t m = 0; m < rows_this_block; ++m) {
        rows.wait_front(WINDOW_STICKS);
        // p = 0
        if (m == 0) {
            tilize_window<
                block_ch_tiles,
                InitUninitMode::InitOnly,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(rows, windows, 1);
        } else {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, 1);
        }
        // p = 1, 2
        for (uint32_t p = 1; p + 1 < TAP_COUNT; ++p) {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, 1);
        }
        // p = 3: release the rest of the tile-row; the block's last tile-row also releases the
        // 3 trailing sticks, so the block pops exactly its slot (32 * rows + 3).
        if (m + 1 == rows_this_block) {
            tilize_window<block_ch_tiles, InitUninitMode::UninitOnly, Keep>(rows, windows, TILE_DIM);
        } else {
            tilize_window<block_ch_tiles, InitUninitMode::Neither, Keep>(rows, windows, TILE_DIM - HALO_ROWS);
        }
    }
}

// conv_silu_block: rows_this_block x block_ch_tiles output tiles. Output tile (m, c) =
// silu(sum_{p=0..3} window[m*4w + p*w + c] * rowbcast(tap_tile[c], row 2p)), the 4 products
// accumulated in one DEST slot in fixed order p = 0..3, SiLU applied in DEST, packed once.
template <uint32_t block_ch_tiles>
void conv_silu_block(DataflowBuffer& windows, DataflowBuffer& outputs, uint32_t rows_this_block) {
    constexpr uint32_t dest_group = compute_kernel_lib::DEST_AUTO_LIMIT;
    const uint32_t window_tiles = TAP_COUNT * block_ch_tiles * rows_this_block;
    windows.wait_front(window_tiles);

    reconfig_data_format(cb_window_tiles, cb_tap_tiles);
    pack_reconfig_data_format(cb_output_tiles);
    mul_bcast_rows_init(cb_window_tiles, cb_tap_tiles);
    // SiLU runs on the packer thread (TRISC2), overlapping the math thread's next group of MACs
    // (op_design.md Perf lamps: measured 197.7 -> 185.9 us on conv1d-004 vs SiLU on the math thread).
    compute_kernel_lib::ActivationInitHelper<KernelActivation::SILU>::init();

    for (uint32_t m = 0; m < rows_this_block; ++m) {
        outputs.reserve_back(block_ch_tiles);
        const uint32_t row_base = m * TAP_COUNT * block_ch_tiles;
        for (uint32_t c0 = 0; c0 < block_ch_tiles; c0 += dest_group) {
            const uint32_t group = (block_ch_tiles - c0 < dest_group) ? (block_ch_tiles - c0) : dest_group;
            tile_regs_acquire();
            for (uint32_t i = 0; i < group; ++i) {
                const uint32_t c = c0 + i;
                for (uint32_t p = 0; p < TAP_COUNT; ++p) {
                    mul_tiles_bcast_rows(
                        cb_window_tiles, cb_tap_tiles, row_base + p * block_ch_tiles + c, c, i, TAP_ROW_STRIDE * p);
                }
            }
            tile_regs_commit();
            // Replaces tile_regs_wait(): math->pack handshake, SiLU on each tile in DEST from the
            // packer thread, SFPU-done stall before pack.
            compute_kernel_lib::apply_activation_from_pack<KernelActivation::SILU>(group);
            for (uint32_t i = 0; i < group; ++i) {
                pack_tile(i, cb_output_tiles);
            }
            tile_regs_release();
        }
        outputs.push_back(block_ch_tiles);
    }
    windows.pop_front(window_tiles);
}

// untilize_rows_block (TILE regime): (rows + extra) tile-rows of input tiles -> row-major.
template <uint32_t block_ch_tiles>
void untilize_rows_block(uint32_t tile_rows) {
    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer untilized_rows(cb_untilized_rows);
    compute_kernel_lib::untilize<block_ch_tiles, cb_input_tiles, cb_untilized_rows>(
        input_tiles, untilized_rows, tile_rows);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr bool tile_input = get_compile_time_arg_val(2) == 1;

    const uint32_t core_seq_start = get_arg_val<uint32_t>(0);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = (core_seq_tiles + block_rows - 1) / block_rows;
    const uint32_t last_block_rows = core_seq_tiles - (num_blocks_this_core - 1) * block_rows;

    if constexpr (tile_input) {
        compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_untilized_rows);
    } else {
        compute_kernel_hw_startup(cb_rows, cb_rows, cb_window_tiles);
    }

    DataflowBuffer rows(cb_rows);
    DataflowBuffer taps(cb_tap_tiles);
    DataflowBuffer windows(cb_window_tiles);
    DataflowBuffer outputs(cb_output_tiles);

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows_this_block = (block_idx + 1 == num_blocks_this_core) ? last_block_rows : block_rows;
        if constexpr (tile_input) {
            // A core whose range starts at t0 > 0 untilizes one extra tile-row on its first block:
            // its rows 29..31 are that block's halo.
            const uint32_t extra_rows = (block_idx == 0 && core_seq_start > 0) ? 1 : 0;
            untilize_rows_block<block_ch_tiles>(rows_this_block + extra_rows);
        }
        windowed_tilize_block<block_ch_tiles>(rows, windows, rows_this_block);
        if (block_idx == 0) {
            taps.wait_front(block_ch_tiles);  // first use: the tap fetch hid behind the first tilize
        }
        conv_silu_block<block_ch_tiles>(windows, outputs, rows_this_block);
    }
    taps.pop_front(block_ch_tiles);
}
