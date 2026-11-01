// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute. Per block of R_b tile-rows (whole row resident):
//   residual_add_block : z = x + res                         -> work_a   (HAS_RESIDUAL)
//   row_mean_block     : mean = sum_w z / W                  -> row_mean (z stays resident)
//   center_block       : c = z - mean (col bcast)            -> work_b
//   row_stats_block    : S = sum_w c * c DEST-accumulated, then in the same DEST window the row
//                        collapse and rsqrt(S / W + eps)     -> inv_std  (c stays resident)
//   normalize_block    : c * inv_std (col bcast)             -> work_a
//   affine_block       : beta (row bcast) + n * gamma         -> output
// affine_block seeds each DEST lane with the beta row broadcast and lets ELWMUL (a DEST
// multiply-accumulate) add n * gamma onto it, so the shift needs no pass of its own.
//
// work_a / work_b alias one allocation; every phase moves whole blocks through each id so page i of
// one is page i of the other. row_mean uses AccumulateViaAdd on tile-aligned W, which never reads
// the scaler slot (bound to cb_gamma_tiles).
//
// row_stats_block uses raw LLK: eltwise_chain's PerRow DEST accumulation packs the accumulated tile at
// row end with no post-accumulation hook, and reduce() reads its input only from a CB, so collapsing
// and finalizing S inside the accumulating DEST window (no fp32 sumsq CB round trip, no copy_tile
// reload, one acquire per row) needs the window written out by hand.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
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
    // compile-time arg 7 (cb_row_sumsq) is unused: S never leaves DEST.
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(8);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(11);
    constexpr bool has_residual = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t width = get_compile_time_arg_val(13);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(14);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(15);
    // Tiles per DEST acquire on the streaming passes; the chain clamps it to DEST capacity.
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(16);

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
    DataflowBuffer output_tiles(cb_output_tiles);
    DataflowBuffer& z_tiles = has_residual ? work_a : input_tiles;

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        // Streaming passes take `dest_lanes` tiles per DEST acquire (one CB handshake per group).
        const auto lane_shape = IterationShape::grid(rows, tensor_w_tiles).block_size(dest_lanes);

        // residual_add_block
        if constexpr (has_residual) {
            MaybeDeviceZoneScope("compute_residual_add");
            add<lane_input(cb_input_tiles), lane_input(cb_residual_tiles), lane_output(cb_row_work_a)>(
                lane_shape, input_tiles, residual_tiles, work_a);
        }

        // row_mean_block (source left resident for center_block)
        {
            MaybeDeviceZoneScope("compute_wait_z");
            z_tiles.wait_front(rows * tensor_w_tiles);
        }
        {
            MaybeDeviceZoneScope("compute_row_mean");
            reduce_mean<
                ReduceDim::REDUCE_ROW,
                cb_z,
                cb_gamma_tiles,
                cb_row_mean,
                ReduceInputPolicy::WaitUpfrontNoPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(rows, tensor_w_tiles), width);
        }

        // center_block
        {
            MaybeDeviceZoneScope("compute_center");
            sub<lane_input(cb_z),
                input(cb_row_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_row_work_b)>(lane_shape, z_tiles, row_mean, work_b);
        }

        // row_stats_block (c stays resident for normalize_block)
        {
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

        // affine_block: DEST lanes seeded with the beta row broadcast; ELWMUL accumulates n * gamma onto
        // them. beta and gamma both unpack through srcB with one page format, and work_a keeps srcA's
        // format from normalize: one srcB reconfig here and none in the chain, whose two elements
        // alternate inits every DEST group and would re-emit an enabled reconfig each time.
        {
            MaybeDeviceZoneScope("compute_affine");
            reconfig_data_format_srcb(cb_inv_std, cb_gamma_tiles);
            eltwise_chain(
                lane_shape,
                UnaryBcast<
                    BroadcastDim::Row,
                    input(cb_beta_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row, same_work_fmt)>{
                    beta_tiles},
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    lane_input(cb_row_work_a, same_work_fmt),
                    input(
                        cb_gamma_tiles,
                        BroadcastDim::Row,
                        WaitPolicy::Upfront,
                        PopPolicy::None,
                        InputTileMapping::Row,
                        same_work_fmt)>{work_a, gamma_tiles},
                // Whole block pushed at the end: the writer sees the block in one burst at the time the
                // split scale/shift pair would have started emitting (focus2 DRAM-contention probe).
                PackTile<output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd)>{output_tiles});
        }
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
