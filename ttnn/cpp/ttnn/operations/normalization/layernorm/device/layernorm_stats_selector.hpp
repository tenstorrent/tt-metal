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
    bool compact_two_pass_fits_in_l1;
};

// Crossover points measured against the centred tile-reduction path.
// FP32 uses shifted two-pass statistics at every width. The tile reducer has
// better ULP behaviour for ordinary parameter-free inputs, but its TF32 intake
// can erase low-order variation below a large shared offset.
// Parameter-free BF16 uses shifted two-pass statistics at every width to avoid
// cancellation when subtracting a large row mean. It is faster through width
// 128 and costs at most 3% in the measured 256-2880 range.
// Gamma-only and beta-only BF16 use tile reductions below width 3232 and
// shifted two-pass statistics at and above it. The paths are neutral at the
// crossover, while two-pass is 16-20% faster in the measured width-4096 cases.
// BFP8 remains on tile reductions except for the calibrated fused residual and
// full-affine case.
constexpr bool use_blackhole_sfpu_stats(const BlackholeStatsSelectorParams& params) {
    const bool has_full_affine = params.has_gamma && params.has_beta;
    // Both the tile reducer and the ordinary tile finaliser read FP32 operands
    // through TF32 SrcA/SrcB. The FP32 two-pass route keeps statistics,
    // residual pre-add, and normalisation in FP32 DEST/SFPU.
    if (params.input_format == tt::DataFormat::Float32) {
        return true;
    }
    if (!params.has_gamma && !params.has_beta && params.input_format == tt::DataFormat::Float16_b) {
        return true;
    }
    if (!params.compact_two_pass_fits_in_l1) {
        return false;
    }
    if (params.fuse_pre_add && has_full_affine) {
        if (params.input_format == tt::DataFormat::Float16_b) {
            return params.padded_width <= 256 || params.padded_width >= 2880;
        }
        return params.input_format == tt::DataFormat::Bfp8_b && params.padded_width >= 2880;
    }

    if (has_full_affine) {
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
    if (arch == tt::ARCH::QUASAR) {
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
    if (arch == tt::ARCH::QUASAR) {
        return StatisticsBackend::TILE_REDUCTION;
    }
    if (arch == tt::ARCH::BLACKHOLE && !is_pre_all_gather && !is_post_all_gather) {
        return StatisticsBackend::TILE_REDUCTION;
    }
    return StatisticsBackend::SFPU_TWO_PASS;
}

}  // namespace ttnn::prim::layernorm
