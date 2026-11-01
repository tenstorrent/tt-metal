// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute. Per block of R_b tile-rows (whole row resident):
//   residual_add_block : z = x + res                         -> work_a   (HAS_RESIDUAL)
//   row_mean_block     : mean = sum_w z / W                  -> row_mean (z stays resident)
//   center_block       : c = z - mean (col bcast)            -> work_b
//   row_sumsq_block    : S = sum_w c * c, DEST-accumulated   -> row_sumsq (c stays resident)
//   row_inv_std_block  : rsqrt(collapse(S) / W + eps)        -> inv_std
//   normalize_block    : c * inv_std (col bcast)             -> work_a
//   affine_scale_block : * gamma (row bcast)                 -> work_b
//   affine_shift_block : + beta (row bcast)                  -> output
// With FUSE_NORMALIZE_AFFINE the last three run as one DEST pass (normalize_affine_block): no fp32
// work-page round trip between them.
//
// work_a / work_b alias one allocation; every phase moves whole blocks through each id so page i of
// one is page i of the other. Both reduces use AccumulateViaAdd on tile-aligned W, which never reads
// the scaler slot (bound to cb_gamma_tiles). The inv_std post-op uses raw SFPU calls because
// reduce()'s post_reduce_op hook takes a DEST-index callable, not chain elements.
//
// normalize_affine_block uses raw LLK: the chain's DestReuseBinary takes no broadcast operand, and
// gamma/beta are row-0 tiles that need a ROW broadcast on SrcB. The LLK supports ROW broadcast with
// DEST_TO_SRCA reuse; the chain can adopt this once DestReuseBinary takes a BroadcastInputSpec.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

using compute_kernel_lib::DataFormatReconfig;

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

// Applies `Op` with a ROW-broadcast tile of `cb_row` (column `col0 + lane`) to DEST lanes [0, n),
// one operand taken from DEST (DEST_TO_SRCA), the broadcast row on SrcB.
template <EltwiseBinaryType Op, uint32_t cb_row>
ALWI void dest_reuse_row_bcast(uint32_t col0, uint32_t n) {
    constexpr auto reuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, reuse>(false, false, cb_row)));
    MATH((llk_math_eltwise_binary_init<Op, BroadcastType::ROW, MATH_FIDELITY, reuse>(cb_row, cb_row, false)));
    for (uint32_t lane = 0; lane < n; ++lane) {
        UNPACK((llk_unpack_A<BroadcastType::ROW, true, reuse>(cb_row, col0 + lane)));
        MATH((llk_math_eltwise_binary<Op, BroadcastType::ROW, DST_ACCUM_MODE, MATH_FIDELITY, reuse>(
            cb_row, cb_row, lane, true)));
    }
}

// out = c * inv_std (col bcast) * gamma (row bcast) + beta (row bcast), DEST-resident between ops.
// c (rows x Wt) and inv_std (rows) are already at the CB fronts; both are popped here.
template <
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t tensor_w_tiles>
ALWI void normalize_affine_block(
    uint32_t rows, DataflowBuffer& centered, DataflowBuffer& inv_std, DataflowBuffer& output) {
    constexpr uint32_t lanes = compute_kernel_lib::DEST_AUTO_LIMIT;
    inv_std.wait_front(rows);
    pack_reconfig_data_format(cb_out);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += lanes) {
            const uint32_t n = (tensor_w_tiles - col0) < lanes ? (tensor_w_tiles - col0) : lanes;
            tile_regs_acquire();
            reconfig_data_format(cb_centered, cb_inv_std);
            mul_bcast_cols_init(cb_centered, cb_inv_std);
            for (uint32_t lane = 0; lane < n; ++lane) {
                mul_tiles_bcast_cols(cb_centered, cb_inv_std, row * tensor_w_tiles + col0 + lane, row, lane);
            }
            reconfig_data_format_srcb(cb_inv_std, cb_gamma);
            dest_reuse_row_bcast<EltwiseBinaryType::ELWMUL, cb_gamma>(col0, n);
            reconfig_data_format_srcb(cb_gamma, cb_beta);
            dest_reuse_row_bcast<EltwiseBinaryType::ELWADD, cb_beta>(col0, n);
            tile_regs_commit();
            output.reserve_back(n);
            tile_regs_wait();
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile(lane, cb_out);
            }
            tile_regs_release();
            output.push_back(n);
        }
    }
    centered.pop_front(rows * tensor_w_tiles);
    inv_std.pop_front(rows);
}

// ---------------------------------------------------------------------------------------------
// fuse_add_mean variants (raw compute API; why the helpers are bypassed is noted per function).
// ---------------------------------------------------------------------------------------------

// Within-tile finalize of a row sum sitting in DST[0]: collapse the 32 lanes (sfpu_reduce ROW) and
// scale by the true 1/W -- the same two calls reduce_mean<AccumulateViaAdd> issues.
ALWI void finalize_row_mean(uint32_t inv_w_bits) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
    mul_unary_tile(0, inv_w_bits);
}

// V1 pack_l1acc: z = x + res, `lanes` tiles per DEST acquire (same as the helper's add<>). The packer
// writes each z lane to work_a AND L1-accumulates it into zsum slot `lane`, so zsum ends up holding the
// elementwise row sum split over `slots` tiles with no extra unpack or math. The add<> chain cannot
// do this: its OutputSpec packs each lane to exactly one CB and has no packer-L1-accumulate mode.
template <
    uint32_t cb_in,
    uint32_t cb_res,
    uint32_t cb_work,
    uint32_t cb_zsum,
    uint32_t Wt,
    uint32_t lanes,
    uint32_t slots>
ALWI void residual_add_pack_l1acc(
    uint32_t rows, DataflowBuffer& input, DataflowBuffer& residual, DataflowBuffer& work, DataflowBuffer& zsum) {
    reconfig_data_format(cb_in, cb_res);
    pack_reconfig_data_format(cb_work);
    add_init(cb_in, cb_res, false);
    zsum.reserve_back(rows * slots);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col0 = 0; col0 < Wt; col0 += lanes) {
            const uint32_t n = (Wt - col0) < lanes ? (Wt - col0) : lanes;
            input.wait_front(n);
            residual.wait_front(n);
            tile_regs_acquire();
            for (uint32_t lane = 0; lane < n; ++lane) {
                add_tiles(cb_in, cb_res, lane, lane, lane);
            }
            tile_regs_commit();
            input.pop_front(n);
            residual.pop_front(n);
            work.reserve_back(n);
            tile_regs_wait();
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile(lane, cb_work);
            }
            // Slot `lane` is first written by group 0 (n == min(lanes, Wt) == slots there): overwrite.
            if (col0 != 0) {
                pack_reconfig_l1_acc(1);
            }
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile<true>(lane, cb_zsum, row * slots + lane);
            }
            if (col0 != 0) {
                pack_reconfig_l1_acc(0);
            }
            tile_regs_release();
            work.push_back(n);
        }
    }
    zsum.push_back(rows * slots);
}

// V4 fpu_acc_lane: `lanes - 1` z lanes plus one partial-sum lane, DST[lanes-1] += x + res via an
// acc_to_dest add_tiles; the partial is L1-accumulated into one zsum tile per row.
template <uint32_t cb_in, uint32_t cb_res, uint32_t cb_work, uint32_t cb_zsum, uint32_t Wt, uint32_t lanes>
ALWI void residual_add_fpu_acc_lane(
    uint32_t rows, DataflowBuffer& input, DataflowBuffer& residual, DataflowBuffer& work, DataflowBuffer& zsum) {
    constexpr uint32_t zl = lanes - 1;
    constexpr uint32_t acc_lane = lanes - 1;
    reconfig_data_format(cb_in, cb_res);
    pack_reconfig_data_format(cb_work);
    zsum.reserve_back(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col0 = 0; col0 < Wt; col0 += zl) {
            const uint32_t n = (Wt - col0) < zl ? (Wt - col0) : zl;
            input.wait_front(n);
            residual.wait_front(n);
            tile_regs_acquire();
            add_init(cb_in, cb_res, false);
            for (uint32_t lane = 0; lane < n; ++lane) {
                add_tiles(cb_in, cb_res, lane, lane, lane);
            }
            add_init(cb_in, cb_res, true);
            for (uint32_t lane = 0; lane < n; ++lane) {
                add_tiles(cb_in, cb_res, lane, lane, acc_lane);
            }
            tile_regs_commit();
            input.pop_front(n);
            residual.pop_front(n);
            work.reserve_back(n);
            tile_regs_wait();
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile(lane, cb_work);
            }
            if (col0 != 0) {
                pack_reconfig_l1_acc(1);
            }
            pack_tile<true>(acc_lane, cb_zsum, row);
            if (col0 != 0) {
                pack_reconfig_l1_acc(0);
            }
            tile_regs_release();
            work.push_back(n);
        }
    }
    zsum.push_back(rows);
}

// V2 stream: the row sum accumulates in DST[0] as the reader's chunks land (cumulative wait_front, no
// pop), so after the last chunk only that chunk's adds plus the collapse remain. Without residual the
// operand pair is (x, x) two tiles per add; with residual it is (x_k, res_k) one column per add, read
// straight from the bf16 input CBs before residual_add pops them. reduce()'s policies are either
// wait-everything-upfront or wait-and-pop; none waits incrementally while leaving the tiles resident.
template <uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t Wt, bool pair_same_cb>
ALWI void row_mean_stream(
    uint32_t rows, DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, uint32_t inv_w_bits) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    sfpu_reduce_init<PoolType::SUM, dst_fmt>();
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t base = row * Wt;
        tile_regs_acquire();
        if constexpr (pair_same_cb) {
            uint32_t k = 0;
            if (Wt & 1u) {
                a.wait_front(base + 1);
                copy_tile_init(cb_a);
                copy_tile(cb_a, base, 0);
                k = 1;
            }
            add_init(cb_a, cb_a, true);
            for (; k < Wt; k += 2) {
                a.wait_front(base + k + 2);
                add_tiles(cb_a, cb_a, base + k, base + k + 1, 0);
            }
        } else {
            add_init(cb_a, cb_b, true);
            for (uint32_t k = 0; k < Wt; ++k) {
                a.wait_front(base + k + 1);
                b.wait_front(base + k + 1);
                add_tiles(cb_a, cb_b, base + k, base + k, 0);
            }
        }
        finalize_row_mean(inv_w_bits);
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
    }
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
    constexpr uint32_t cb_row_sumsq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(8);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(11);
    constexpr bool has_residual = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t width = get_compile_time_arg_val(13);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(14);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(15);
    constexpr bool fuse_normalize_affine = get_compile_time_arg_val(16) != 0;
    // Tiles per DEST acquire on the streaming passes; the chain clamps it to DEST capacity.
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(17);
    constexpr uint32_t mean_variant = get_compile_time_arg_val(18);
    constexpr uint32_t cb_zsum = get_compile_time_arg_val(19);
    constexpr uint32_t zsum_slots = get_compile_time_arg_val(20);
    constexpr uint32_t clamped_lanes =
        dest_lanes < compute_kernel_lib::DEST_AUTO_LIMIT ? dest_lanes : compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr bool res_pack_l1acc = has_residual && (mean_variant == 1 || mean_variant == 3);
    constexpr bool res_fpu_acc_lane = has_residual && mean_variant == 4;
    constexpr bool res_stream = has_residual && mean_variant == 2;
    constexpr bool nores_stream = !has_residual && (mean_variant == 2 || mean_variant == 3);

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
    DataflowBuffer row_sumsq(cb_row_sumsq);
    DataflowBuffer inv_std(cb_inv_std);
    DataflowBuffer output_tiles(cb_output_tiles);
    [[maybe_unused]] DataflowBuffer zsum(cb_zsum);
    DataflowBuffer& z_tiles = has_residual ? work_a : input_tiles;

    constexpr InputSpec centered_resident =
        input(cb_row_work_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    using SquareAccumulate =
        BinaryFpu<BinaryFpuOp::Mul, centered_resident, centered_resident, Dst::D0, DestAccumulation::PerRow>;
    using SumsqPack = PackTile<output(
        cb_row_sumsq,
        ReservePolicy::PerOuter,
        PushPolicy::PerOuter,
        DataFormatReconfig::Enabled,
        TileAddressing::Direct,
        DestAccumulation::PerRow)>;

    // The REDUCE_ROW result lives in column 0 (faces 0 and 2), and normalize reads only that column
    // through its col broadcast, so the SFPU post-op runs on VectorMode::C: half the faces.
    auto finalize_inv_std = [=](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
            dst_idx,
            VectorMode::C,
            inv_w_bits));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
            dst_idx,
            VectorMode::C,
            epsilon_bits));
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX, 8, DST_ACCUM_MODE, false),
            dst_idx,
            VectorMode::C));
    };

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const auto block_shape = IterationShape::grid(rows, tensor_w_tiles);
        // Streaming passes take `dest_lanes` tiles per DEST acquire (one CB handshake per group).
        const auto lane_shape = IterationShape::grid(rows, tensor_w_tiles).block_size(dest_lanes);

        if constexpr (res_pack_l1acc || res_fpu_acc_lane) {
            {
                MaybeDeviceZoneScope("compute_residual_add");
                if constexpr (res_pack_l1acc) {
                    residual_add_pack_l1acc<
                        cb_input_tiles,
                        cb_residual_tiles,
                        cb_row_work_a,
                        cb_zsum,
                        tensor_w_tiles,
                        clamped_lanes,
                        zsum_slots>(rows, input_tiles, residual_tiles, work_a, zsum);
                } else {
                    residual_add_fpu_acc_lane<
                        cb_input_tiles,
                        cb_residual_tiles,
                        cb_row_work_a,
                        cb_zsum,
                        tensor_w_tiles,
                        clamped_lanes>(rows, input_tiles, residual_tiles, work_a, zsum);
                }
            }
            {
                MaybeDeviceZoneScope("compute_row_mean");
                reduce_mean<
                    ReduceDim::REDUCE_ROW,
                    cb_zsum,
                    cb_gamma_tiles,
                    cb_row_mean,
                    ReduceInputPolicy::BulkWaitBulkPop,
                    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(rows, zsum_slots), width);
            }
        } else if constexpr (res_stream || nores_stream) {
            {
                MaybeDeviceZoneScope("compute_row_mean");
                if constexpr (res_stream) {
                    row_mean_stream<cb_input_tiles, cb_residual_tiles, cb_row_mean, tensor_w_tiles, false>(
                        rows, input_tiles, residual_tiles, row_mean, inv_w_bits);
                } else {
                    row_mean_stream<cb_input_tiles, cb_input_tiles, cb_row_mean, tensor_w_tiles, true>(
                        rows, input_tiles, input_tiles, row_mean, inv_w_bits);
                }
            }
            if constexpr (has_residual) {
                MaybeDeviceZoneScope("compute_residual_add");
                add<lane_input(cb_input_tiles), lane_input(cb_residual_tiles), lane_output(cb_row_work_a)>(
                    lane_shape, input_tiles, residual_tiles, work_a);
            }
        } else {
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
                if constexpr (mean_variant == 5) {
                    // V5 col_scale: SUM + 1/W on VectorMode::C only (the REDUCE_ROW result is column 0).
                    reduce<
                        PoolType::SUM,
                        ReduceDim::REDUCE_ROW,
                        cb_z,
                        cb_gamma_tiles,
                        cb_row_mean,
                        ReduceInputPolicy::WaitUpfrontNoPop,
                        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ReduceFp32Mode::Fast,
                        ReduceAlgorithm::AccumulateViaAdd>(
                        ReduceInputBlockShape::of(rows, tensor_w_tiles),
                        ReduceInputMemoryLayout::contiguous(),
                        NoAccumulation{},
                        [=](uint32_t dst_idx) {
                            MATH(SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                DST_ACCUM_MODE,
                                calculate_binop_with_scalar,
                                (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
                                dst_idx,
                                VectorMode::C,
                                inv_w_bits));
                        });
                } else {
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
            }
        }

        // center_block
        {
            MaybeDeviceZoneScope("compute_center");
            sub<lane_input(cb_z),
                input(cb_row_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_row_work_b)>(lane_shape, z_tiles, row_mean, work_b);
        }

        // row_sumsq_block (c stays resident for normalize_block)
        {
            MaybeDeviceZoneScope("compute_row_sumsq");
            work_b.wait_front(rows * tensor_w_tiles);
            eltwise_chain(block_shape, SquareAccumulate{work_b, work_b}, SumsqPack{row_sumsq});
        }

        // row_inv_std_block
        {
            MaybeDeviceZoneScope("compute_inv_std");
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_row_sumsq,
                cb_gamma_tiles,
                cb_inv_std,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::AccumulateViaAdd,
                ReduceWithinTile::Collapse>(
                ReduceInputBlockShape::of(rows, 1),
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                finalize_inv_std);
        }

        if constexpr (fuse_normalize_affine) {
            gamma_tiles.wait_front(tensor_w_tiles);
            beta_tiles.wait_front(tensor_w_tiles);
            MaybeDeviceZoneScope("compute_normalize_affine");
            normalize_affine_block<
                cb_row_work_b,
                cb_inv_std,
                cb_gamma_tiles,
                cb_beta_tiles,
                cb_output_tiles,
                tensor_w_tiles>(rows, work_b, inv_std, output_tiles);
        } else {
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
                    input(
                        cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
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
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
