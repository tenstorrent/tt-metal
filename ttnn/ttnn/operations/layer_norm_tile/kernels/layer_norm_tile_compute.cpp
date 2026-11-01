// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute. Per block of R_b tile-rows, per row, per G-tile stats window (G = stats_chunk_tiles,
// a whole number of the reader's C-tile chunks, or Wt):
//   residual_add_block : z = x + res                         -> work_a   (HAS_RESIDUAL)
//   row_stats_window   : as the window's tiles land, S1 += z (and S2 += z * z when one_pass) in DEST;
//                        between windows the raw sums ride cb_stat_partial (fp32 tagged UnpackToDestFp32,
//                        lossless reload). Last window of the row: row collapse, then
//                        mean = S1 / W                         -> row_mean
//                        inv_std = rsqrt(S2 / W - mean^2 + eps) -> inv_std  (one_pass only)
//   center_block       : c = z - mean (col bcast)            -> work_b
//   row_stats_block    : (two-pass only) S = sum_w c * c, collapse, rsqrt(S / W + eps) -> inv_std
//   normalize_block    : c * inv_std (col bcast)             -> work_a
//   affine_scale_block : * gamma (row bcast)                 -> work_b
//   affine_shift_block : + beta (row bcast)                  -> output
//
// work_a / work_b alias one allocation; every phase after the stats moves whole blocks through each id so
// page i of one is page i of the other. The source of z stays resident (not popped) until center_block.
//
// row_stats_window / row_stats_block use raw LLK. reduce() reads a resident block only after waiting all
// of it, or pops what it streams; neither keeps z resident while accumulating it chunk by chunk as the
// reader pushes, and it cannot carry the sum of squares alongside the sum. eltwise_chain's PerRow DEST
// accumulation has no post-accumulation hook, so collapsing and finalizing inside the accumulating window
// needs the window written out by hand.
//
// one_pass (host-gated to fp32 DEST + HiFi4): z is read through SrcA at TF32, z * z is exact under HiFi4
// and accumulates in fp32 DEST, so S2 / W - mean^2 keeps ~1e-3 absolute error at |mean| ~ 100, std 1.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// rsqrt(x * scale + addend) on the SFPU loads holding tile column 0 only: a load covers 4 rows x 8
// columns (dest address bit 1 picks even / odd columns) and dst_reg[k] is address 2k, so column 0 is
// dst_reg[0], [2], [4], [6]. Half the rsqrt work of a VectorMode::C pass; normalize's col broadcast
// reads only column 0. Same rsqrt body as rsqrt_tile.
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, uint32_t scale_bits>
inline void calculate_add_rsqrt_col0(uint32_t addend_bits) {
    const sfpi::vFloat scale = Converter::as_float(scale_bits);
    const sfpi::vFloat addend = Converter::as_float(addend_bits);
#pragma GCC unroll 4
    for (int group = 0; group < 4; ++group) {
        sfpi::vFloat y =
            _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(sfpi::dst_reg[2 * group] * scale + addend);
        if constexpr (!fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[2 * group] = y;
    }
}

// Column 0 of DEST tiles 0 (S1) and 1 (S2, one_pass) after the row collapse: tile 0 <- mean = S1 * scale,
// tile 1 <- rsqrt(max(S2 * scale - mean^2, 0) + addend). Tile stride is 32 dst_reg rows.
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, bool one_pass, uint32_t scale_bits>
inline void calculate_row_mean_inv_std_col0(uint32_t addend_bits) {
    constexpr uint32_t tile_rows = 32;
    const sfpi::vFloat scale = Converter::as_float(scale_bits);
#pragma GCC unroll 4
    for (int group = 0; group < 4; ++group) {
        sfpi::vFloat mean = sfpi::dst_reg[2 * group] * scale;
        if constexpr (!fp32_dest_acc_en) {
            mean = sfpi::convert<sfpi::vFloat16b>(mean, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[2 * group] = mean;
        if constexpr (one_pass) {
            const sfpi::vFloat addend = Converter::as_float(addend_bits);
            sfpi::vFloat var = sfpi::dst_reg[tile_rows + 2 * group] * scale - mean * mean;
            v_if(var < 0.0f) { var = 0.0f; }
            v_endif;
            sfpi::dst_reg[tile_rows + 2 * group] = _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(var + addend);
        }
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {

using compute_kernel_lib::DataFormatReconfig;

// c (rows x Wt, resident at the cb_centered front, not popped) -> inv_std (one tile per row, column 0
// valid). Per row one DEST window: S = sum_w c * c (mul, acc_to_dest), sfpu_reduce ROW collapse, then
// rsqrt(S * 1/W + eps) on column 0.
template <
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void row_stats_block(uint32_t rows, DataflowBuffer& centered, DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    centered.wait_front(rows * tensor_w_tiles);
    reconfig_data_format(cb_centered, cb_centered);
    pack_reconfig_data_format(cb_inv_std);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t base = row * tensor_w_tiles;
        tile_regs_acquire();
        mul_init(cb_centered, cb_centered, true);  // a freshly acquired DEST reads 0: the first mul seeds S
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            mul_tiles(cb_centered, cb_centered, base + col, base + col, 0);
        }
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_add_rsqrt_col0,
            (APPROX, DST_ACCUM_MODE, inv_w_bits),
            0,
            VectorMode::C,
            epsilon_bits));
        tile_regs_commit();
        inv_std.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_inv_std);
        tile_regs_release();
        inv_std.push_back(1);
    }
}

// One stats window: tiles [col0, col0 + G) of `row` at the z front (waited cumulatively, not popped).
// DEST 0 accumulates S1 (pairwise acc add), DEST 1 S2 (acc mul) when one_pass. A row's first window starts
// from a fresh (zero) DEST; later ones reload the raw sums from cb_stat_partial. Odd G occurs only as
// G = Wt (single window per row): its first tile seeds DEST 0 by copy.
template <
    uint32_t cb_z,
    uint32_t cb_stat_partial,
    uint32_t cb_row_mean,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t window_tiles,
    bool one_pass,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void row_stats_window(
    uint32_t row,
    uint32_t col0,
    DataflowBuffer& z,
    DataflowBuffer& partial,
    DataflowBuffer& row_mean,
    DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    constexpr uint32_t num_sums = one_pass ? 2 : 1;
    static_assert(window_tiles % 2 == 0 || window_tiles == tensor_w_tiles, "odd windows only as one per row");
    const uint32_t base = row * tensor_w_tiles + col0;
    const bool first = col0 == 0;
    const bool last = col0 + window_tiles == tensor_w_tiles;

    z.wait_front(base + window_tiles);
    reconfig_data_format(cb_z, cb_z);
    tile_regs_acquire();
    uint32_t tile = 0;
    if (!first) {
        partial.wait_front(num_sums);
        reconfig_data_format_srca(cb_z, cb_stat_partial);
        copy_init(cb_stat_partial);
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            copy_tile(cb_stat_partial, sum, sum);
        }
        reconfig_data_format_srca(cb_stat_partial, cb_z);
        partial.pop_front(num_sums);
    } else if constexpr (window_tiles % 2 != 0) {
        copy_init(cb_z);
        copy_tile(cb_z, base, 0);
        tile = 1;
    }
    add_init(cb_z, cb_z, true);
    for (; tile < window_tiles; tile += 2) {
        add_tiles(cb_z, cb_z, base + tile, base + tile + 1, 0);
    }
    if constexpr (one_pass) {
        mul_init(cb_z, cb_z, true);
        for (uint32_t col = 0; col < window_tiles; ++col) {
            mul_tiles(cb_z, cb_z, base + col, base + col, 1);
        }
    }
    if (last) {
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(sum, 1, 1);
        }
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_row_mean_inv_std_col0,
            (APPROX, DST_ACCUM_MODE, one_pass, inv_w_bits),
            0,
            VectorMode::C,
            epsilon_bits));
    }
    tile_regs_commit();
    if (last) {
        pack_reconfig_data_format(cb_row_mean);
        row_mean.reserve_back(1);
        if constexpr (one_pass) {
            inv_std.reserve_back(1);
        }
        tile_regs_wait();
        pack_tile(0, cb_row_mean);
        if constexpr (one_pass) {
            pack_tile(1, cb_inv_std);
        }
        tile_regs_release();
        row_mean.push_back(1);
        if constexpr (one_pass) {
            inv_std.push_back(1);
        }
    } else {
        pack_reconfig_data_format(cb_stat_partial);
        partial.reserve_back(num_sums);
        tile_regs_wait();
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            pack_tile(sum, cb_stat_partial);
        }
        tile_regs_release();
        partial.push_back(num_sums);
    }
}

// Streaming-pass operands: one CB handshake per IterationShape::block_size group of DEST lanes.
constexpr compute_kernel_lib::InputSpec lane_input(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block, reconfig);
}

constexpr compute_kernel_lib::OutputSpec lane_output(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return output(cb, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize, reconfig);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_row_work_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_row_work_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_row_mean = get_compile_time_arg_val(6);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(7);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(10);
    constexpr bool has_residual = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(12);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(13);
    // Tiles per DEST acquire on the streaming passes; the chain clamps it to DEST capacity.
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(14);
    // Tiles per row-stats window: a multiple of the reader's chunk (stats start while the row lands) or Wt.
    constexpr uint32_t stats_window_tiles = get_compile_time_arg_val(15);
    constexpr bool one_pass = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t cb_stat_partial = get_compile_time_arg_val(17);
    static_assert(tensor_w_tiles % stats_window_tiles == 0, "a stats window never straddles rows");

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;

    // Reconfig is dropped only where the format cannot change: work_a / work_b alias one allocation
    // with one page format, and gamma / beta share the I/O format (l1_ledger.md).
    constexpr auto same_work_fmt = DataFormatReconfig::Disabled;

    // Source of the mean and the centering: z = x + res when present, else x itself.
    constexpr uint32_t cb_z = has_residual ? cb_row_work_a : cb_input_tiles;

    compute_kernel_hw_startup(cb_input_tiles, cb_gamma_tiles, cb_output_tiles);

    DataflowBuffer input_tiles(cb_input_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer beta_tiles(cb_beta_tiles);
    DataflowBuffer work_a(cb_row_work_a);
    DataflowBuffer work_b(cb_row_work_b);
    DataflowBuffer row_mean(cb_row_mean);
    DataflowBuffer inv_std(cb_inv_std);
    DataflowBuffer stat_partial(cb_stat_partial);
    DataflowBuffer output_tiles(cb_output_tiles);
    DataflowBuffer& z_tiles = has_residual ? work_a : input_tiles;

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        // Streaming passes take `dest_lanes` tiles per DEST acquire (one CB handshake per group).
        const auto lane_shape = IterationShape::grid(rows, tensor_w_tiles).block_size(dest_lanes);

        // residual_add_block + row_stats_window, interleaved per window so the stats run as tiles land
        {
            MaybeDeviceZoneScope("compute_add_and_stats");
            const auto window_shape = IterationShape::grid(1, stats_window_tiles).block_size(dest_lanes);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += stats_window_tiles) {
                    if constexpr (has_residual) {
                        add<lane_input(cb_input_tiles), lane_input(cb_residual_tiles), lane_output(cb_row_work_a)>(
                            window_shape, input_tiles, residual_tiles, work_a);
                    }
                    row_stats_window<
                        cb_z,
                        cb_stat_partial,
                        cb_row_mean,
                        cb_inv_std,
                        tensor_w_tiles,
                        stats_window_tiles,
                        one_pass,
                        inv_w_bits,
                        epsilon_bits>(row, col0, z_tiles, stat_partial, row_mean, inv_std);
                }
            }
        }

        // center_block
        {
            MaybeDeviceZoneScope("compute_center");
            sub<lane_input(cb_z),
                input(cb_row_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_row_work_b)>(lane_shape, z_tiles, row_mean, work_b);
        }

        // row_stats_block (two-pass; c stays resident for normalize_block)
        if constexpr (!one_pass) {
            MaybeDeviceZoneScope("compute_row_stats");
            row_stats_block<cb_row_work_b, cb_inv_std, tensor_w_tiles, inv_w_bits, epsilon_bits>(rows, work_b, inv_std);
        }

        // normalize_block
        {
            MaybeDeviceZoneScope("compute_normalize");
            mul<lane_input(cb_row_work_b),
                input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_row_work_a)>(lane_shape, work_b, inv_std, work_a);
        }
        {
            MaybeDeviceZoneScope("compute_wait_affine");
            gamma_tiles.wait_front(tensor_w_tiles);
            beta_tiles.wait_front(tensor_w_tiles);
        }

        // affine_scale_block
        {
            MaybeDeviceZoneScope("compute_affine_scale");
            mul<lane_input(cb_row_work_a, same_work_fmt),
                input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                lane_output(cb_row_work_b, same_work_fmt)>(lane_shape, work_a, gamma_tiles, work_b);
        }

        // affine_shift_block
        {
            MaybeDeviceZoneScope("compute_affine_shift");
            add<lane_input(cb_row_work_b, same_work_fmt),
                input(
                    cb_beta_tiles,
                    BroadcastDim::Row,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    InputTileMapping::Row,
                    same_work_fmt),
                lane_output(cb_output_tiles)>(lane_shape, work_b, beta_tiles, output_tiles);
        }
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
