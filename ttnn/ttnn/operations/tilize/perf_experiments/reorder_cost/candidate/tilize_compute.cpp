// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute — block operation `tilize_block`.
//
// One compute_kernel_lib::tilize call per block (block_height_tiles tile-rows x width tiles):
// init / uninit / reconfig once per block (Phase 0: once per core). The helper owns all CB
// wait/pop/reserve/push (symmetric tile-sized pages, block width per tile-row). The ragged last
// column block is a second instantiation on its own CB pair.
//
// Fp32Mode::Lossless: a 32-bit input CB is tagged UnpackToDestFp32 by the host (with
// fp32_dest_acc_en), so fp32 takes the bit-exact standard tilize path instead of fast tilize
// (tf32 truncation). The mode has no effect on non-fp32 input formats.
//
// bfloat4_b output (bfp4_round): the packer TRUNCATES every 16-element shared-exponent group to a
// 3-bit magnitude. Before the tilize, each window of round_tile_rows tile-rows is streamed through
// `unary<Bfp4Round>` (copy page -> DEST, SFPU round-to-nearest-even onto the group's bfp4 grid,
// pack bf16) into cb_bfp4_rounded, and the tilize reads that. A 16-element group is 16 contiguous
// elements of a stick, i.e. one 16-wide face row of the (non-tilized) page in DEST, so the group
// is visible to the SFPU without the tilize. On-grid values survive the packer's truncation
// exactly: the output equals the host's round-to-nearest bfp4 conversion. The custom SFPU element
// exists because no kernel_lib / LLK op rounds onto a per-row shared-exponent grid.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// x rounded to nearest-even on the bfp4 grid of its group: step 2^(E - 127 - 2) where E (biased)
// is the group's max exponent, magnitude clamped to 7 steps (host converter saturation). In units
// of the step |x| is in [0, 8); |x| / step + 2^23 lands in [2^23, 2^23 + 8) where the fp32 ulp is
// 1, so the MAD rounds it to an integer, half to even. Clamp there, then one more MAD scales back
// and removes the 2^23 (both exact: powers of two times small integers). Elements below half a
// step become 0. Groups with E < 3 (denormal range; the 2^(2 - Eu) inverse would overflow) -> 0.
constexpr float BFP4_RNE_MAGIC = 8388608.0f;  // 2^23
constexpr float BFP4_MAX_STEPS = 7.0f;        // 3-bit magnitude
sfpi_inline sfpi::vFloat bfp4_round_value(
    sfpi::vFloat x, sfpi::vFloat step, sfpi::vFloat inv_step, sfpi::vFloat neg_magic_step) {
    sfpi::vFloat q = sfpi::abs(x) * inv_step + BFP4_RNE_MAGIC;
    q = sfpi::min(q, BFP4_RNE_MAGIC + BFP4_MAX_STEPS);
    return sfpi::copysgn(q * step + neg_magic_step, x);
}

// One face (16 rows x 16 cols) per call; VectorMode::RC calls it once per face. A row band is
// dst_reg[0] (rows r..r+3, cols 0..7) + dst_reg[1] (same rows, cols 8..15); within a register a
// face row is one 8-lane sub-vector, so a row's max is a lane-wise max of the two halves followed
// by a max over the sub-vector (rotate-by-1/2/4 within the sub-vector).
inline void calculate_bfp4_round() {
    constexpr int ROW_BANDS_PER_FACE = 4;
#pragma GCC unroll 4
    for (int band = 0; band < ROW_BANDS_PER_FACE; ++band) {
        const sfpi::vFloat lo = sfpi::dst_reg[0];
        const sfpi::vFloat hi = sfpi::dst_reg[1];
        sfpi::vFloat m = sfpi::max(sfpi::abs(lo), sfpi::abs(hi));
        sfpi::vFloat r = sfpi::vFloat(sfpi::subvec_shflror1(m));
        m = sfpi::max(m, r);
        r = sfpi::vFloat(sfpi::subvec_shflror1(m));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        m = sfpi::max(m, r);
        r = sfpi::vFloat(sfpi::subvec_shflror1(m));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        r = sfpi::vFloat(sfpi::subvec_shflror1(r));
        m = sfpi::max(m, r);
        const sfpi::vInt emax_b = sfpi::exexp(m, sfpi::ExponentMode::Biased);
        const sfpi::vFloat step = sfpi::setexp(sfpi::vFloat(1.0f), emax_b - 2);          // 2^(Eu - 2)
        sfpi::vFloat inv_step = sfpi::setexp(sfpi::vFloat(1.0f), (127 + 129) - emax_b);  // 2^(2 - Eu)
        v_if(emax_b < 3) { inv_step = 0.0f; }
        v_endif;
        const sfpi::vFloat neg_magic_step = step * -BFP4_RNE_MAGIC;
        sfpi::dst_reg[0] = bfp4_round_value(lo, step, inv_step, neg_magic_step);
        sfpi::dst_reg[1] = bfp4_round_value(hi, step, inv_step, neg_magic_step);
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {
// Chain element: round DEST slot D0 onto the bfp4 grid (see calculate_bfp4_round).
struct Bfp4Round : compute_kernel_lib::UnaryOp<Bfp4Round, compute_kernel_lib::Dst::D0> {
    static ALWI void init() { MATH(SFPU_UNARY_INIT(unused)); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_bfp4_round, slot_offset, VectorMode::RC));
    }
};

template <uint32_t block_width_tiles, uint32_t cb_in, uint32_t cb_out>
ALWI void tilize_block(uint32_t rows) {
    using namespace compute_kernel_lib::tilize_config;
    compute_kernel_lib::tilize<
        block_width_tiles,
        cb_in,
        cb_out,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
        Fp32Mode::Lossless>(rows);
}

// One row block of `rows` tile-rows; with bfp4_round, rounded first in windows of
// round_tile_rows tile-rows (the round pass and the tilize share the TRISCs and alternate).
template <
    uint32_t block_width_tiles,
    uint32_t cb_in,
    uint32_t cb_out,
    bool bfp4_round,
    uint32_t cb_bfp4_rounded,
    uint32_t round_tile_rows>
ALWI void tilize_rows(uint32_t rows) {
    if constexpr (!bfp4_round) {
        // Occupancy: the helper waits on cb_in / reserves cb_out per tile-row inside this zone.
        MaybeDeviceZoneScope("compute_tilize");
        tilize_block<block_width_tiles, cb_in, cb_out>(rows);
    } else {
        using namespace compute_kernel_lib;
        for (uint32_t r = 0; r < rows; r += round_tile_rows) {
            const uint32_t n = (rows - r < round_tile_rows) ? rows - r : round_tile_rows;
            {
                MaybeDeviceZoneScope("compute_bfp4_round");
                unary<Bfp4Round, input(cb_in), output(cb_bfp4_rounded)>(IterationShape::tiles(n * block_width_tiles));
            }
            {
                MaybeDeviceZoneScope("compute_tilize");
                tilize_block<block_width_tiles, cb_bfp4_rounded, cb_out>(n);
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_sticks_tail = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tiles_out_tail = get_compile_time_arg_val(3);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(6);
    constexpr bool has_tail = get_compile_time_arg_val(7) != 0;
    constexpr bool bfp4_round = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t cb_bfp4_rounded = get_compile_time_arg_val(9);
    constexpr uint32_t round_tile_rows = get_compile_time_arg_val(10);
    constexpr uint32_t num_full_col_blocks = num_col_blocks - (has_tail ? 1 : 0);

    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_rm_sticks, cb_tiles_out);

    for (uint32_t col_block = 0; col_block < num_full_col_blocks; ++col_block) {
        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rows =
                (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
            tilize_rows<block_width_tiles, cb_rm_sticks, cb_tiles_out, bfp4_round, cb_bfp4_rounded, round_tile_rows>(
                rows);
        }
    }

    if constexpr (has_tail) {
        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rows =
                (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
            tilize_rows<
                last_block_width_tiles,
                cb_rm_sticks_tail,
                cb_tiles_out_tail,
                bfp4_round,
                cb_bfp4_rounded,
                round_tile_rows>(rows);
        }
    }
}
