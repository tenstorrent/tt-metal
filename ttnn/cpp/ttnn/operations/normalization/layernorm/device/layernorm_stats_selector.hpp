// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::prim::layernorm {

enum class StatisticsBackend : std::uint8_t { TILE_REDUCTION, SFPU_TWO_PASS };

struct BlackholeStatsSelectorParams {
    tt::DataFormat input_format;
    std::uint32_t padded_width;
    bool fuse_pre_add;
    bool has_gamma;
    bool has_beta;
    std::uint32_t num_tile_rows;
    std::uint32_t active_cores;
    std::uint32_t available_cores;
    bool compact_two_pass_fits_in_l1;
};

// Crossover points measured against the centred tile-reduction path.
// Parameter-free FP32 remains on tile reductions for its better ULP behaviour.
// Parameter-free BF16 uses shifted two-pass statistics at every width to avoid
// cancellation when subtracting a large row mean. It is faster through width
// 128 and costs at most 3% in the measured 256-2880 range.
// BFP8 remains on tile reductions except for the calibrated fused residual and
// full-affine case.
constexpr bool use_blackhole_sfpu_stats(const BlackholeStatsSelectorParams& params) {
    const bool has_full_affine = params.has_gamma && params.has_beta;
    // The tile reducer forms the residual sum through TF32 SrcA/SrcB and can
    // erase variation below a large shared FP32 offset. The specialised
    // two-pass residual path keeps pre-add and normalisation in FP32 SFPU.
    if (params.input_format == tt::DataFormat::Float32 && params.fuse_pre_add && !params.has_gamma &&
        !params.has_beta && !params.compact_two_pass_fits_in_l1) {
        return true;
    }
    if (!params.has_gamma && !params.has_beta && params.input_format == tt::DataFormat::Float16_b) {
        return true;
    }
    // This large fused shape cannot use the compact allocation, but retaining
    // the post-add row and multicasting affine parameters makes SFPU two-pass
    // substantially faster than the tile reducer on a full Blackhole grid.
    const bool calibrated_fp32_residual_replay = params.input_format == tt::DataFormat::Float32 &&
                                                 params.fuse_pre_add && has_full_affine &&
                                                 params.padded_width == 2880 && params.num_tile_rows == 32 &&
                                                 params.active_cores == 32 && params.available_cores >= 100;
    if (calibrated_fp32_residual_replay) {
        return true;
    }
    if (!params.compact_two_pass_fits_in_l1) {
        return false;
    }

    // FP32 crossover measurements were taken with enough rows to occupy the
    // selected grid. Keep small or restricted-core workloads on the tile path
    // until independently calibrated.
    const bool fp32_parallelism_calibrated = params.active_cores == params.available_cores &&
                                             params.available_cores >= 100 &&
                                             params.num_tile_rows >= params.active_cores;
    if (params.fuse_pre_add && has_full_affine) {
        if (params.input_format == tt::DataFormat::Float32) {
            return fp32_parallelism_calibrated && params.padded_width >= 2048;
        }
        if (params.input_format == tt::DataFormat::Float16_b) {
            return params.padded_width <= 256 || params.padded_width >= 2880;
        }
        return params.input_format == tt::DataFormat::Bfp8_b && params.padded_width >= 2880;
    }

    if (has_full_affine) {
        if (params.input_format == tt::DataFormat::Float32) {
            return fp32_parallelism_calibrated && params.padded_width >= 2880;
        }
        return params.input_format == tt::DataFormat::Float16_b;
    }

    return params.input_format == tt::DataFormat::Float16_b && params.padded_width >= 3232;
}

constexpr StatisticsBackend select_interleaved_statistics_backend(
    bool requested_use_welford,
    tt::ARCH arch,
    bool rms_norm,
    bool input_is_row_major,
    bool fp32_dest_acc_en,
    const BlackholeStatsSelectorParams& blackhole_params) {
    if (!requested_use_welford) {
        return StatisticsBackend::TILE_REDUCTION;
    }
    // RMSNorm does not execute the Welford calculation, but its existing
    // Welford-configured route has distinct kernel and CB requirements.
    if (arch != tt::ARCH::BLACKHOLE || rms_norm || input_is_row_major) {
        return StatisticsBackend::SFPU_TWO_PASS;
    }
    return fp32_dest_acc_en && use_blackhole_sfpu_stats(blackhole_params) ? StatisticsBackend::SFPU_TWO_PASS
                                                                          : StatisticsBackend::TILE_REDUCTION;
}

constexpr StatisticsBackend select_sharded_statistics_backend(
    bool requested_use_welford, tt::ARCH arch, bool is_pre_all_gather, bool is_post_all_gather) {
    if (!requested_use_welford) {
        return StatisticsBackend::TILE_REDUCTION;
    }
    if (arch == tt::ARCH::BLACKHOLE && !is_pre_all_gather && !is_post_all_gather) {
        return StatisticsBackend::TILE_REDUCTION;
    }
    return StatisticsBackend::SFPU_TWO_PASS;
}

}  // namespace ttnn::prim::layernorm
