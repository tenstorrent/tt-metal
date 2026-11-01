// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// three_pass_restructure bake-off copy of layer_norm_tile_compute.cpp. LN3P_MODE:
//   0/1 : the op's kernel verbatim (1 = FUSE_NORMALIZE_AFFINE, set by the host).
//   2   : fused output stage from c (c * inv_std * gamma + beta per DEST lane group) with gamma/beta
//         copied once per core into work-format CBs, so the lane loop carries no data-format reconfig.
//   3   : three-pass restructure. (B) c = z - mean per lane in DEST, sum of c^2 accumulated by the
//         SFPU into DEST lane 0 across the whole row (one acquire per row, c never packed);
//         (C) ((z - mean) * inv_std) * gamma + beta per DEST lane group, packed straight to output.
// Raw LLK: the chain's DestReuseBinary has no broadcast operand (COL inv_std, ROW gamma/beta), and
// no helper squares a DEST-resident tile into a DEST accumulator (the SFPU MAD below).
//
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
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
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

#ifndef LN3P_MODE
#define LN3P_MODE 0
#endif

// dest lane 0 (+)= sum_{k=1..N} lane_k^2, fp32 SFPU MAD; one face per call (8 SFPU rows).
#ifdef TRISC_MATH
template <uint32_t N, bool FIRST>
inline void square_accumulate_face(uint32_t, uint32_t, uint32_t) {
    constexpr uint32_t tile_rows = 32;  // sfpi rows per DEST tile
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat c1 = sfpi::dst_reg[tile_rows];
        sfpi::vFloat acc;
        if constexpr (FIRST) {
            acc = c1 * c1;
        } else {
            acc = c1 * c1 + sfpi::dst_reg[0];
        }
#pragma GCC unroll 4
        for (uint32_t k = 2; k <= N; ++k) {
            sfpi::vFloat c = sfpi::dst_reg[k * tile_rows];
            acc = c * c + acc;
        }
        sfpi::dst_reg[0] = acc;
        sfpi::dst_reg++;
    }
}
#endif

template <bool FIRST>
ALWI void square_accumulate(uint32_t n) {
    switch (n) {
        case 1:
            MATH((_llk_math_eltwise_binary_sfpu_params_(square_accumulate_face<1, FIRST>, 0, 0, 0, VectorMode::RC)));
            break;
        case 2:
            MATH((_llk_math_eltwise_binary_sfpu_params_(square_accumulate_face<2, FIRST>, 0, 0, 0, VectorMode::RC)));
            break;
        default:
            MATH((_llk_math_eltwise_binary_sfpu_params_(square_accumulate_face<3, FIRST>, 0, 0, 0, VectorMode::RC)));
            break;
    }
    // The next FPU op overwrites lanes 1..n: wait for the SFPU to finish reading them.
    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1)));
}

// Applies `Op` with a `B`-broadcast tile `tile0 + lane` (or `tile0` when `same_tile`) of `cb_b` to DEST
// lanes [0, n), the other operand taken from DEST (DEST_TO_SRCA).
template <EltwiseBinaryType Op, BroadcastType B, uint32_t cb_b, bool same_tile>
ALWI void dest_reuse_bcast(uint32_t tile0, uint32_t n) {
    constexpr auto reuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    UNPACK((llk_unpack_A_init<B, true, reuse>(false, false, cb_b)));
    MATH((llk_math_eltwise_binary_init<Op, B, MATH_FIDELITY, reuse>(cb_b, cb_b, false)));
    for (uint32_t lane = 0; lane < n; ++lane) {
        UNPACK((llk_unpack_A<B, true, reuse>(cb_b, same_tile ? tile0 : tile0 + lane)));
        MATH((llk_math_eltwise_binary<Op, B, DST_ACCUM_MODE, MATH_FIDELITY, reuse>(cb_b, cb_b, lane, true)));
    }
}

// Copies `count` affine tiles into a work-format CB (once per core) so the fused output stage's SrcB
// stays in one format.
template <uint32_t cb_in, uint32_t cb_out>
ALWI void affine_to_work(uint32_t count) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_in);
    for (uint32_t col0 = 0; col0 < count; col0 += 4) {
        const uint32_t n = (count - col0) < 4 ? (count - col0) : 4;
        tile_regs_acquire();
        for (uint32_t lane = 0; lane < n; ++lane) {
            copy_tile(cb_in, col0 + lane, lane);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, n);
        tile_regs_wait();
        for (uint32_t lane = 0; lane < n; ++lane) {
            pack_tile(lane, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, n);
    }
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

    constexpr uint32_t cb_gamma_copy = get_compile_time_arg_val(18);
    constexpr uint32_t cb_beta_copy = get_compile_time_arg_val(19);
    // bf16 DEST: work format == affine format, no copy needed.
    constexpr bool affine_copy = LN3P_MODE >= 2 && DST_ACCUM_MODE;
    constexpr uint32_t cb_gamma_w = affine_copy ? cb_gamma_copy : cb_gamma_tiles;
    constexpr uint32_t cb_beta_w = affine_copy ? cb_beta_copy : cb_beta_tiles;
    constexpr uint32_t lanes = DST_ACCUM_MODE ? 4 : 8;
    constexpr uint32_t c_lanes = 3;  // mode 3 pass B: lane 0 accumulates, lanes 1..3 hold c

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
    DataflowBuffer& z_tiles = has_residual ? work_a : input_tiles;

    constexpr InputSpec centered_resident =
        input(cb_row_work_b, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    using SquareAccumulate [[maybe_unused]] =
        BinaryFpu<BinaryFpuOp::Mul, centered_resident, centered_resident, Dst::D0, DestAccumulation::PerRow>;
    using SumsqPack [[maybe_unused]] = PackTile<output(
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
        [[maybe_unused]] const auto block_shape = IterationShape::grid(rows, tensor_w_tiles);
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

#if LN3P_MODE == 3
        {
            MaybeDeviceZoneScope("compute_center_sumsq");
            row_mean.wait_front(rows);
            pack_reconfig_data_format(cb_row_sumsq);
            for (uint32_t row = 0; row < rows; ++row) {
                tile_regs_acquire();
                add_binary_tile_init();
                reconfig_data_format(cb_z, cb_row_mean);
                sub_bcast_cols_init(cb_z, cb_row_mean);
                for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += c_lanes) {
                    const uint32_t n = (tensor_w_tiles - col0) < c_lanes ? (tensor_w_tiles - col0) : c_lanes;
                    for (uint32_t k = 0; k < n; ++k) {
                        sub_tiles_bcast_cols(cb_z, cb_row_mean, row * tensor_w_tiles + col0 + k, row, 1 + k);
                    }
                    if (col0 == 0) {
                        square_accumulate<true>(n);
                    } else {
                        square_accumulate<false>(n);
                    }
                }
                tile_regs_commit();
                row_sumsq.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, cb_row_sumsq);
                tile_regs_release();
                row_sumsq.push_back(1);
            }
        }
#else
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
#endif

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

        if constexpr (affine_copy) {
            if (block_row == 0) {
                MaybeDeviceZoneScope("compute_affine_to_work");
                gamma_tiles.wait_front(tensor_w_tiles);
                beta_tiles.wait_front(tensor_w_tiles);
                affine_to_work<cb_gamma_tiles, cb_gamma_w>(tensor_w_tiles);
                affine_to_work<cb_beta_tiles, cb_beta_w>(tensor_w_tiles);
                cb_wait_front(cb_gamma_w, tensor_w_tiles);
                cb_wait_front(cb_beta_w, tensor_w_tiles);
            }
        }
#if LN3P_MODE >= 2
        {
            gamma_tiles.wait_front(tensor_w_tiles);
            beta_tiles.wait_front(tensor_w_tiles);
            cb_wait_front(cb_gamma_w, tensor_w_tiles);
            cb_wait_front(cb_beta_w, tensor_w_tiles);
            MaybeDeviceZoneScope("compute_output_fused");
            inv_std.wait_front(rows);
            pack_reconfig_data_format(cb_output_tiles);
            // Mode 2 reads c (work_b); mode 3 recomputes c = z - mean from z.
            constexpr uint32_t cb_src = LN3P_MODE == 3 ? cb_z : cb_row_work_b;
            if constexpr (LN3P_MODE == 2) {
                work_b.wait_front(rows * tensor_w_tiles);
            }
            reconfig_data_format(cb_src, cb_inv_std);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += lanes) {
                    const uint32_t n = (tensor_w_tiles - col0) < lanes ? (tensor_w_tiles - col0) : lanes;
                    tile_regs_acquire();
                    if constexpr (LN3P_MODE == 3) {
                        // Without a residual z is the bf16 input; SrcA must be back in the work format
                        // before DEST is moved into it, or the dest-reuse ops see c rounded to bf16.
                        constexpr bool z_in_work_fmt = has_residual || !DST_ACCUM_MODE;
                        if constexpr (!z_in_work_fmt) {
                            reconfig_data_format_srca(cb_inv_std, cb_z);
                        }
                        sub_bcast_cols_init(cb_z, cb_row_mean);
                        for (uint32_t lane = 0; lane < n; ++lane) {
                            sub_tiles_bcast_cols(cb_z, cb_row_mean, row * tensor_w_tiles + col0 + lane, row, lane);
                        }
                        if constexpr (!z_in_work_fmt) {
                            reconfig_data_format_srca(cb_z, cb_inv_std);
                        }
                        dest_reuse_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL, cb_inv_std, true>(row, n);
                    } else {
                        mul_bcast_cols_init(cb_row_work_b, cb_inv_std);
                        for (uint32_t lane = 0; lane < n; ++lane) {
                            mul_tiles_bcast_cols(
                                cb_row_work_b, cb_inv_std, row * tensor_w_tiles + col0 + lane, row, lane);
                        }
                    }
                    dest_reuse_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, cb_gamma_w, false>(col0, n);
                    dest_reuse_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW, cb_beta_w, false>(col0, n);
                    tile_regs_commit();
                    output_tiles.reserve_back(n);
                    tile_regs_wait();
                    for (uint32_t lane = 0; lane < n; ++lane) {
                        pack_tile(lane, cb_output_tiles);
                    }
                    tile_regs_release();
                    output_tiles.push_back(n);
                }
            }
            if constexpr (LN3P_MODE == 3) {
                z_tiles.pop_front(rows * tensor_w_tiles);
                row_mean.pop_front(rows);
            } else {
                work_b.pop_front(rows * tensor_w_tiles);
            }
            inv_std.pop_front(rows);
        }
#else
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
#endif
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
    if constexpr (affine_copy) {
        cb_pop_front(cb_gamma_w, tensor_w_tiles);
        cb_pop_front(cb_beta_w, tensor_w_tiles);
    }
}
