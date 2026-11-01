// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// normalize_affine bake-off: out = c * inv_std (col bcast) * gamma (row bcast) + beta (row bcast).
// c sits at the front of cb_wb (fp32), inv_std at the front of cb_inv (fp32, column 0 valid), gamma / beta
// are Wt row-0 tiles already waited. cb_wa aliases cb_wb (op layout). Every variant pops c and inv_std.
//
//   0 BASELINE   three chain passes: wb -> wa (normalize), wa -> wb (scale), wb -> out (shift)
//   1 PARKED     the op's parked raw fused pass (dest-reuse ROW bcast), unchanged
//   2 SEED       normalize pass, then ONE chain pass: DEST seeded with beta (UnaryBcast Row), then
//                ELWMUL wa * gamma (Row) accumulates onto it -> beta + n * gamma. Helper-only.
//   3 PARKED_LEAN parked fused pass with the per-group srcA reconfig and gamma->beta reconfig dropped
//   4 SEED_RAW   variant 2 in raw LLK (same instruction stream minus chain bookkeeping)
//   5 SEED_INTERLEAVED  SEED_RAW's two passes interleaved per DEST group (normalize group -> wa, then
//                beta seed + gamma MAC on that group -> out): output streams after the first group instead
//                of after the whole normalize pass.
//   6 SEED_PIPELINED  variant 5 software-pipelined by one group (N0 N1 S0 N2 S1 ... S_last): the seed group
//                never waits on the pack it depends on, output still streams from the second group.
//
// Variants 2/4 exploit that ELWMUL always multiply-accumulates into DEST (every fidelity phase is a MAC;
// only the acquire-time clear zeroes DEST), so a DEST lane pre-loaded with beta needs no separate add pass.

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace na {

using compute_kernel_lib::DataFormatReconfig;

constexpr uint32_t BASELINE = 0;
constexpr uint32_t PARKED = 1;
constexpr uint32_t SEED = 2;
constexpr uint32_t PARKED_LEAN = 3;
constexpr uint32_t SEED_RAW = 4;
constexpr uint32_t SEED_INTERLEAVED = 5;
constexpr uint32_t SEED_PIPELINED = 6;
// Diagnostic single-op passes (bench only, output values are not the LN result): per-op-tile cost.
constexpr uint32_t DIAG_MUL_COL = 10;
constexpr uint32_t DIAG_MUL_ROW = 11;
constexpr uint32_t DIAG_ADD_ROW = 12;
constexpr uint32_t DIAG_COPY = 13;
constexpr uint32_t DIAG_BCAST_ROW = 14;
constexpr uint32_t DIAG_SEED_2XINIT = 15;  // seed_raw with every per-group init issued twice

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

template <
    bool lean,
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t tensor_w_tiles>
ALWI void fused_dest_reuse(uint32_t rows, DataflowBuffer& centered, DataflowBuffer& inv_std, DataflowBuffer& output) {
    constexpr uint32_t lanes = compute_kernel_lib::DEST_AUTO_LIMIT;
    inv_std.wait_front(rows);
    pack_reconfig_data_format(cb_out);
    if constexpr (lean) {
        reconfig_data_format(cb_centered, cb_inv_std);
    }
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += lanes) {
            const uint32_t n = (tensor_w_tiles - col0) < lanes ? (tensor_w_tiles - col0) : lanes;
            tile_regs_acquire();
            if constexpr (lean) {
                if (col0 != 0 || row != 0) {
                    reconfig_data_format_srcb(cb_beta, cb_inv_std);
                }
            } else {
                reconfig_data_format(cb_centered, cb_inv_std);
            }
            mul_bcast_cols_init(cb_centered, cb_inv_std);
            for (uint32_t lane = 0; lane < n; ++lane) {
                mul_tiles_bcast_cols(cb_centered, cb_inv_std, row * tensor_w_tiles + col0 + lane, row, lane);
            }
            reconfig_data_format_srcb(cb_inv_std, cb_gamma);
            dest_reuse_row_bcast<EltwiseBinaryType::ELWMUL, cb_gamma>(col0, n);
            if constexpr (!lean) {
                reconfig_data_format_srcb(cb_gamma, cb_beta);
            }
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

// Raw SEED pass: wa (n, fp32) x gamma (Row) MAC onto DEST pre-loaded with beta (Row). wa is consumed per group.
template <
    uint32_t cb_n,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t tensor_w_tiles,
    uint32_t init_reps = 1>
ALWI void seed_raw(uint32_t rows, uint32_t lanes, DataflowBuffer& normalized, DataflowBuffer& output) {
    // srcB carries gamma and beta (same page format): one srcB reconfig per pass, none per group.
    reconfig_data_format(cb_n, cb_gamma);
    pack_reconfig_data_format(cb_out);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += lanes) {
            const uint32_t n = (tensor_w_tiles - col0) < lanes ? (tensor_w_tiles - col0) : lanes;
            normalized.wait_front(n);
            tile_regs_acquire();
            for (uint32_t k = 0; k < init_reps; ++k) {
                unary_bcast_init<BroadcastType::ROW>(cb_beta);
            }
            for (uint32_t lane = 0; lane < n; ++lane) {
                unary_bcast<BroadcastType::ROW>(cb_beta, col0 + lane, lane);
            }
            for (uint32_t k = 0; k < init_reps; ++k) {
                mul_bcast_rows_init(cb_n, cb_gamma);
            }
            for (uint32_t lane = 0; lane < n; ++lane) {
                mul_tiles_bcast_rows(cb_n, cb_gamma, lane, col0 + lane, lane);
            }
            tile_regs_commit();
            output.reserve_back(n);
            tile_regs_wait();
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile(lane, cb_out);
            }
            tile_regs_release();
            output.push_back(n);
            normalized.pop_front(n);
        }
    }
}

// Interleaved SEED: per DEST group, c * inv_std (Col) -> wa, then beta seed + wa * gamma (Row) MAC -> out.
template <
    uint32_t cb_c,
    uint32_t cb_inv_std,
    uint32_t cb_n,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t tensor_w_tiles>
ALWI void seed_interleaved(
    uint32_t rows,
    uint32_t lanes,
    DataflowBuffer& centered,
    DataflowBuffer& inv_std,
    DataflowBuffer& normalized,
    DataflowBuffer& output) {
    inv_std.wait_front(rows);
    reconfig_data_format(cb_c, cb_inv_std);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += lanes) {
            const uint32_t n = (tensor_w_tiles - col0) < lanes ? (tensor_w_tiles - col0) : lanes;
            centered.wait_front(n);
            tile_regs_acquire();
            if (row != 0 || col0 != 0) {
                reconfig_data_format_srcb(cb_gamma, cb_inv_std);
            }
            mul_bcast_cols_init(cb_c, cb_inv_std);
            for (uint32_t lane = 0; lane < n; ++lane) {
                mul_tiles_bcast_cols(cb_c, cb_inv_std, lane, row, lane);
            }
            tile_regs_commit();
            normalized.reserve_back(n);
            tile_regs_wait();
            pack_reconfig_data_format(cb_n);
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile(lane, cb_n);
            }
            tile_regs_release();
            normalized.push_back(n);
            centered.pop_front(n);

            normalized.wait_front(n);
            tile_regs_acquire();
            reconfig_data_format_srcb(cb_inv_std, cb_gamma);
            unary_bcast_init<BroadcastType::ROW>(cb_beta);
            for (uint32_t lane = 0; lane < n; ++lane) {
                unary_bcast<BroadcastType::ROW>(cb_beta, col0 + lane, lane);
            }
            mul_bcast_rows_init(cb_n, cb_gamma);
            for (uint32_t lane = 0; lane < n; ++lane) {
                mul_tiles_bcast_rows(cb_n, cb_gamma, lane, col0 + lane, lane);
            }
            tile_regs_commit();
            output.reserve_back(n);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t lane = 0; lane < n; ++lane) {
                pack_tile(lane, cb_out);
            }
            tile_regs_release();
            output.push_back(n);
            normalized.pop_front(n);
        }
    }
    inv_std.pop_front(rows);
}

template <uint32_t cb_c, uint32_t cb_inv_std, uint32_t cb_n>
ALWI void normalize_group(uint32_t row, uint32_t n, DataflowBuffer& centered, DataflowBuffer& normalized) {
    centered.wait_front(n);
    tile_regs_acquire();
    mul_bcast_cols_init(cb_c, cb_inv_std);
    for (uint32_t lane = 0; lane < n; ++lane) {
        mul_tiles_bcast_cols(cb_c, cb_inv_std, lane, row, lane);
    }
    tile_regs_commit();
    normalized.reserve_back(n);
    tile_regs_wait();
    pack_reconfig_data_format(cb_n);
    for (uint32_t lane = 0; lane < n; ++lane) {
        pack_tile(lane, cb_n);
    }
    tile_regs_release();
    normalized.push_back(n);
    centered.pop_front(n);
}

template <uint32_t cb_n, uint32_t cb_gamma, uint32_t cb_beta, uint32_t cb_out>
ALWI void seed_group(uint32_t col0, uint32_t n, DataflowBuffer& normalized, DataflowBuffer& output) {
    normalized.wait_front(n);
    tile_regs_acquire();
    unary_bcast_init<BroadcastType::ROW>(cb_beta);
    for (uint32_t lane = 0; lane < n; ++lane) {
        unary_bcast<BroadcastType::ROW>(cb_beta, col0 + lane, lane);
    }
    mul_bcast_rows_init(cb_n, cb_gamma);
    for (uint32_t lane = 0; lane < n; ++lane) {
        mul_tiles_bcast_rows(cb_n, cb_gamma, lane, col0 + lane, lane);
    }
    tile_regs_commit();
    output.reserve_back(n);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    for (uint32_t lane = 0; lane < n; ++lane) {
        pack_tile(lane, cb_out);
    }
    tile_regs_release();
    output.push_back(n);
    normalized.pop_front(n);
}

// One-group-lag pipeline over a row: N0, N1, S0, N2, S1, ..., S_last. Within a row the seed of group g reads
// wa pages packed one group earlier, so no acquire waits on the pack it depends on.
template <
    uint32_t cb_c,
    uint32_t cb_inv_std,
    uint32_t cb_n,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t tensor_w_tiles>
ALWI void seed_pipelined(
    uint32_t rows,
    uint32_t lanes,
    DataflowBuffer& centered,
    DataflowBuffer& inv_std,
    DataflowBuffer& normalized,
    DataflowBuffer& output) {
    inv_std.wait_front(rows);
    reconfig_data_format(cb_c, cb_inv_std);
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t prev_col0 = 0;
        uint32_t prev_n = 0;
        for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += lanes) {
            const uint32_t n = (tensor_w_tiles - col0) < lanes ? (tensor_w_tiles - col0) : lanes;
            if (prev_n != 0) {
                reconfig_data_format_srcb(cb_gamma, cb_inv_std);
            }
            normalize_group<cb_c, cb_inv_std, cb_n>(row, n, centered, normalized);
            if (prev_n != 0) {
                reconfig_data_format_srcb(cb_inv_std, cb_gamma);
                seed_group<cb_n, cb_gamma, cb_beta, cb_out>(prev_col0, prev_n, normalized, output);
            }
            prev_col0 = col0;
            prev_n = n;
        }
        reconfig_data_format_srcb(cb_inv_std, cb_gamma);
        seed_group<cb_n, cb_gamma, cb_beta, cb_out>(prev_col0, prev_n, normalized, output);
        reconfig_data_format_srcb(cb_gamma, cb_inv_std);
    }
    inv_std.pop_front(rows);
}

template <
    uint32_t variant,
    uint32_t cb_wa,
    uint32_t cb_wb,
    uint32_t cb_inv_std,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_out,
    uint32_t tensor_w_tiles,
    uint32_t dest_lanes>
ALWI void normalize_affine(
    uint32_t rows,
    DataflowBuffer& work_a,
    DataflowBuffer& work_b,
    DataflowBuffer& inv_std,
    DataflowBuffer& gamma_tiles,
    DataflowBuffer& beta_tiles,
    DataflowBuffer& output_tiles) {
    using namespace compute_kernel_lib;
    constexpr auto same_work_fmt = DataFormatReconfig::Disabled;
    const auto lane_shape = IterationShape::grid(rows, tensor_w_tiles).block_size(dest_lanes);

    auto wait_affine = [&]() {
        MaybeDeviceZoneScope("compute_wait_affine");
        gamma_tiles.wait_front(tensor_w_tiles);
        beta_tiles.wait_front(tensor_w_tiles);
    };

    if constexpr (variant == DIAG_SEED_2XINIT) {
        mul<lane_input(cb_wb),
            input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_wa)>(lane_shape, work_b, inv_std, work_a);
        constexpr uint32_t lanes = dest_lanes < DEST_AUTO_LIMIT ? dest_lanes : DEST_AUTO_LIMIT;
        seed_raw<cb_wa, cb_gamma, cb_beta, cb_out, tensor_w_tiles, 2>(rows, lanes, work_a, output_tiles);
        return;
    } else if constexpr (variant >= DIAG_MUL_COL) {
        if constexpr (variant == DIAG_MUL_COL) {
            mul<lane_input(cb_wb),
                input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_output(cb_out)>(lane_shape, work_b, inv_std, output_tiles);
            return;
        } else if constexpr (variant == DIAG_MUL_ROW) {
            mul<lane_input(cb_wb),
                input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                lane_output(cb_out)>(lane_shape, work_b, gamma_tiles, output_tiles);
        } else if constexpr (variant == DIAG_ADD_ROW) {
            add<lane_input(cb_wb),
                input(cb_beta, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                lane_output(cb_out)>(lane_shape, work_b, beta_tiles, output_tiles);
        } else if constexpr (variant == DIAG_COPY) {
            copy<lane_input(cb_wb), lane_output(cb_out)>(lane_shape, work_b, output_tiles);
        } else if constexpr (variant == DIAG_BCAST_ROW) {
            work_b.wait_front(rows * tensor_w_tiles);
            unary_bcast<
                BroadcastDim::Row,
                input(cb_beta, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                lane_output(cb_out)>(lane_shape, beta_tiles, output_tiles);
            work_b.pop_front(rows * tensor_w_tiles);
        }
        inv_std.wait_front(rows);
        inv_std.pop_front(rows);
        return;
    }

    if constexpr (variant == SEED_INTERLEAVED) {
        wait_affine();
        MaybeDeviceZoneScope("compute_normalize_affine");
        constexpr uint32_t lanes = dest_lanes < DEST_AUTO_LIMIT ? dest_lanes : DEST_AUTO_LIMIT;
        seed_interleaved<cb_wb, cb_inv_std, cb_wa, cb_gamma, cb_beta, cb_out, tensor_w_tiles>(
            rows, lanes, work_b, inv_std, work_a, output_tiles);
        return;
    }

    if constexpr (variant == SEED_PIPELINED) {
        wait_affine();
        MaybeDeviceZoneScope("compute_normalize_affine");
        constexpr uint32_t lanes = dest_lanes < DEST_AUTO_LIMIT ? dest_lanes : DEST_AUTO_LIMIT;
        seed_pipelined<cb_wb, cb_inv_std, cb_wa, cb_gamma, cb_beta, cb_out, tensor_w_tiles>(
            rows, lanes, work_b, inv_std, work_a, output_tiles);
        return;
    }

    if constexpr (variant == PARKED || variant == PARKED_LEAN) {
        wait_affine();
        MaybeDeviceZoneScope("compute_normalize_affine");
        fused_dest_reuse<variant == PARKED_LEAN, cb_wb, cb_inv_std, cb_gamma, cb_beta, cb_out, tensor_w_tiles>(
            rows, work_b, inv_std, output_tiles);
        return;
    }

    {
        MaybeDeviceZoneScope("compute_normalize");
        mul<lane_input(cb_wb),
            input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_wa)>(lane_shape, work_b, inv_std, work_a);
    }
    wait_affine();

    if constexpr (variant == BASELINE) {
        {
            MaybeDeviceZoneScope("compute_affine_scale");
            mul<lane_input(cb_wa, same_work_fmt),
                input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
                lane_output(cb_wb, same_work_fmt)>(lane_shape, work_a, gamma_tiles, work_b);
        }
        {
            MaybeDeviceZoneScope("compute_affine_shift");
            add<lane_input(cb_wb, same_work_fmt),
                input(
                    cb_beta,
                    BroadcastDim::Row,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    InputTileMapping::Row,
                    same_work_fmt),
                lane_output(cb_out)>(lane_shape, work_b, beta_tiles, output_tiles);
        }
    } else if constexpr (variant == SEED) {
        MaybeDeviceZoneScope("compute_affine_seed");
        // beta and gamma both unpack through srcB and share one page format, and wa shares wb's format (srcA
        // after normalize): one srcB reconfig ahead of the chain, none inside it. An enabled reconfig on any
        // of the three is re-emitted every DEST group because the two elements alternate inits.
        reconfig_data_format_srcb(cb_inv_std, cb_gamma);
        eltwise_chain(
            lane_shape,
            UnaryBcast<
                BroadcastDim::Row,
                input(cb_beta, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row, same_work_fmt)>{beta_tiles},
            BinaryFpu<
                BinaryFpuOp::Mul,
                lane_input(cb_wa, same_work_fmt),
                input(
                    cb_gamma,
                    BroadcastDim::Row,
                    WaitPolicy::Upfront,
                    PopPolicy::None,
                    InputTileMapping::Row,
                    same_work_fmt)>{work_a, gamma_tiles},
            PackTile<lane_output(cb_out)>{output_tiles});
    } else if constexpr (variant == SEED_RAW) {
        MaybeDeviceZoneScope("compute_affine_seed");
        constexpr uint32_t lanes = dest_lanes < DEST_AUTO_LIMIT ? dest_lanes : DEST_AUTO_LIMIT;
        seed_raw<cb_wa, cb_gamma, cb_beta, cb_out, tensor_w_tiles>(rows, lanes, work_a, output_tiles);
    }
}

}  // namespace na
