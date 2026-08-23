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
};

// Crossover points measured against the centred tile-reduction path.
// Parameter-free FP32 remains on tile reductions for its better ULP behaviour.
// BFP8 also remains on tile reductions because its SFPU crossover depends on
// affine-finalisation optimisations that are outside this branch.
constexpr bool use_blackhole_sfpu_stats(const BlackholeStatsSelectorParams& params) {
    const bool has_full_affine = params.has_gamma && params.has_beta;
    if (params.fuse_pre_add && has_full_affine) {
        if (params.input_format == tt::DataFormat::Float32) {
            return params.padded_width >= 2048;
        }
        if (params.input_format == tt::DataFormat::Float16_b) {
            return params.padded_width <= 256 || params.padded_width >= 2880;
        }
        return false;
    }

    if (has_full_affine) {
        if (params.input_format == tt::DataFormat::Float32) {
            return params.padded_width >= 2880;
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
