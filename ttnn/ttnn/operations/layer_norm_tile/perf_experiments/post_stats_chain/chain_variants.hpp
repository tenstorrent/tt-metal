// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// post_stats_chain bake-off: the resident whole-row chain center -> row_stats_block -> normalize ->
// affine_scale -> affine_shift (layer_norm_tile_compute.cpp, two-pass path), selected by CHAIN_VARIANT.
//   0 baseline   : the op's code at 9daa4c30ecf.
//   1 cum_wait   : row_stats_block waits c cumulatively per DEST-lane group (c * c starts while center's
//                  packer still lands the row) instead of the whole row up front.
//   2 gamma_first: cum_wait, and the gamma multiply moves ahead of the inv_std multiply:
//                  out = ((c * gamma) * inv_std) + beta. The c * gamma pass does not depend on inv_std, so it
//                  runs while row_stats' tail (SFPU collapse + rsqrt, pack, unpack of inv_std) drains,
//                  hiding the stats -> normalize bubble. Same pass count, same CB traffic.

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef CHAIN_VARIANT
#define CHAIN_VARIANT 0
#endif

#ifdef CHAIN_DIAG
#define CHAIN_DIAG_ZONE(name) DeviceZoneScopedN(name)
#else
#define CHAIN_DIAG_ZONE(name)
#endif

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// Verbatim from the op.
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

namespace post_stats {

using compute_kernel_lib::DataFormatReconfig;

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

// Verbatim from the op.
template <
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits,
    uint32_t wait_step = 0>
ALWI void row_stats_block(uint32_t rows, DataflowBuffer& centered, DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    if constexpr (wait_step == 0) {
        centered.wait_front(rows * tensor_w_tiles);
    }
    reconfig_data_format(cb_centered, cb_centered);
    pack_reconfig_data_format(cb_inv_std);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t base = row * tensor_w_tiles;
        tile_regs_acquire();
        mul_init(cb_centered, cb_centered, true);
        {
            CHAIN_DIAG_ZONE("diag_sumsq");
            for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
                if constexpr (wait_step != 0) {
                    // cumulative: c tiles are consumed as center's packer lands them (never popped here)
                    if (col % wait_step == 0) {
                        const uint32_t upto = col + wait_step < tensor_w_tiles ? col + wait_step : tensor_w_tiles;
                        centered.wait_front(base + upto);
                    }
                }
                mul_tiles(cb_centered, cb_centered, base + col, base + col, 0);
            }
        }
        {
            CHAIN_DIAG_ZONE("diag_collapse");
            sfpu_reduce_init<PoolType::SUM, dst_fmt>();
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
        }
        {
            CHAIN_DIAG_ZONE("diag_rsqrt");
            rsqrt_tile_init();
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_add_rsqrt_col0,
                (APPROX, DST_ACCUM_MODE, inv_w_bits),
                0,
                VectorMode::C,
                epsilon_bits));
        }
        tile_regs_commit();
        inv_std.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_inv_std);
        tile_regs_release();
        inv_std.push_back(1);
    }
}

template <
    uint32_t cb_z,
    uint32_t cb_mean,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_work_a,
    uint32_t cb_work_b,
    uint32_t cb_inv_std,
    uint32_t cb_out,
    uint32_t cb_aux,
    uint32_t Wt,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits,
    uint32_t dest_lanes>
ALWI void chain(uint32_t rows) {
    using namespace compute_kernel_lib;
    constexpr auto same_work_fmt = DataFormatReconfig::Disabled;
    DataflowBuffer z_tiles(cb_z), row_mean(cb_mean), gamma_tiles(cb_gamma), beta_tiles(cb_beta);
    DataflowBuffer work_a(cb_work_a), work_b(cb_work_b), inv_std(cb_inv_std), output_tiles(cb_out);
    const auto lane_shape = IterationShape::grid(rows, Wt).block_size(dest_lanes);

#if CHAIN_VARIANT == 0
    {
        MaybeDeviceZoneScope("compute_center");
        sub<lane_input(cb_z),
            input(cb_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_work_b)>(lane_shape, z_tiles, row_mean, work_b);
    }
    {
        MaybeDeviceZoneScope("compute_row_stats");
        row_stats_block<cb_work_b, cb_inv_std, Wt, inv_w_bits, epsilon_bits>(rows, work_b, inv_std);
    }
    {
        MaybeDeviceZoneScope("compute_normalize");
        mul<lane_input(cb_work_b),
            input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_work_a)>(lane_shape, work_b, inv_std, work_a);
    }
    {
        MaybeDeviceZoneScope("compute_affine_scale");
        mul<lane_input(cb_work_a, same_work_fmt),
            input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            lane_output(cb_work_b, same_work_fmt)>(lane_shape, work_a, gamma_tiles, work_b);
    }
    {
        MaybeDeviceZoneScope("compute_affine_shift");
        add<lane_input(cb_work_b, same_work_fmt),
            input(
                cb_beta, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row, same_work_fmt),
            lane_output(cb_out)>(lane_shape, work_b, beta_tiles, output_tiles);
    }
#elif CHAIN_VARIANT == 1
    {
        MaybeDeviceZoneScope("compute_center");
        sub<lane_input(cb_z),
            input(cb_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_work_b)>(lane_shape, z_tiles, row_mean, work_b);
    }
    {
        MaybeDeviceZoneScope("compute_row_stats");
        row_stats_block<cb_work_b, cb_inv_std, Wt, inv_w_bits, epsilon_bits, dest_lanes>(rows, work_b, inv_std);
    }
    {
        MaybeDeviceZoneScope("compute_normalize");
        mul<lane_input(cb_work_b),
            input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_work_a)>(lane_shape, work_b, inv_std, work_a);
    }
    {
        MaybeDeviceZoneScope("compute_affine_scale");
        mul<lane_input(cb_work_a, same_work_fmt),
            input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            lane_output(cb_work_b, same_work_fmt)>(lane_shape, work_a, gamma_tiles, work_b);
    }
    {
        MaybeDeviceZoneScope("compute_affine_shift");
        add<lane_input(cb_work_b, same_work_fmt),
            input(
                cb_beta, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row, same_work_fmt),
            lane_output(cb_out)>(lane_shape, work_b, beta_tiles, output_tiles);
    }
#elif CHAIN_VARIANT == 2
    {
        MaybeDeviceZoneScope("compute_center");
        sub<lane_input(cb_z),
            input(cb_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_work_b)>(lane_shape, z_tiles, row_mean, work_b);
    }
    {
        MaybeDeviceZoneScope("compute_row_stats");
        row_stats_block<cb_work_b, cb_inv_std, Wt, inv_w_bits, epsilon_bits, dest_lanes>(rows, work_b, inv_std);
    }
    {
        MaybeDeviceZoneScope("compute_affine_scale");
        mul<lane_input(cb_work_b),
            input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            lane_output(cb_work_a, same_work_fmt)>(lane_shape, work_b, gamma_tiles, work_a);
    }
    {
        MaybeDeviceZoneScope("compute_normalize");
        mul<lane_input(cb_work_a, same_work_fmt),
            input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            lane_output(cb_work_b, same_work_fmt)>(lane_shape, work_a, inv_std, work_b);
    }
    {
        MaybeDeviceZoneScope("compute_affine_shift");
        add<lane_input(cb_work_b),
            input(cb_beta, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            lane_output(cb_out)>(lane_shape, work_b, beta_tiles, output_tiles);
    }
#endif
}

}  // namespace post_stats
