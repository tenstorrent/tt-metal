// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// The one compute configuration for every chunk_gated_delta_rule program (mono, phased, fused). The
// kernels keep the recurrent state and the WY inverse in fp32 DEST at HiFi4 and nothing else has been
// validated, so other arithmetic is rejected; dst_full_sync_en passes through, other knobs are unused.
inline tt::tt_metal::ComputeConfigDescriptor gdn_compute_config(const DeviceComputeKernelConfig& cfg) {
    using tt::tt_metal::MathFidelity;
    TT_FATAL(
        cfg.math_fidelity == MathFidelity::HiFi4 && cfg.fp32_dest_acc_en && !cfg.math_approx_mode,
        "chunk_gated_delta_rule runs HiFi4 with fp32 destination accumulation and no approx mode on every "
        "path (its recurrent state and WY inverse are fp32); compute_kernel_config asked for math_fidelity {} "
        "(HiFi4 = {}), fp32_dest_acc_en {}, math_approx_mode {}",
        static_cast<int>(cfg.math_fidelity),
        static_cast<int>(MathFidelity::HiFi4),
        cfg.fp32_dest_acc_en,
        cfg.math_approx_mode);
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = cfg.math_fidelity,
        .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
        .dst_full_sync_en = cfg.dst_full_sync_en,
        .math_approx_mode = cfg.math_approx_mode};
}

// WY-inverse method of the prep compute (the `tinv` attribute of the prep and fused prims): the op's
// ChunkGdnWyInverse with AUTO resolved for this device and chunk size.
//   HORNER    : invert_block — quadrant split, two 15-term Horner inverses and an exact off-diagonal on
//               the matrix engine. Runs on every architecture at every chunk size.
//   SFPU_FP32 : one SFPU forward-substitution solve reading negN as fp32 in place
//               (triangle_solve_tile, api/compute/triangle_solve.h). Blackhole-only, chunk_size == 32.
enum class GdnTinv : uint32_t { HORNER = 0, SFPU_FP32 = 1 };

inline bool gdn_tinv_sfpu_supported(uint32_t chunk_size, const Tensor& any_input) {
    return chunk_size == tt::constants::TILE_HEIGHT && any_input.device()->arch() == tt::ARCH::BLACKHOLE;
}

inline GdnTinv gdn_tinv_resolve(
    ttnn::transformer::ChunkGdnWyInverse wy_inverse, uint32_t chunk_size, const Tensor& any_input) {
    using ttnn::transformer::ChunkGdnWyInverse;
    switch (wy_inverse) {
        case ChunkGdnWyInverse::HORNER: return GdnTinv::HORNER;
        case ChunkGdnWyInverse::FORWARD_SUBSTITUTION: return GdnTinv::SFPU_FP32;  // validate FATALs if unsupported
        case ChunkGdnWyInverse::AUTO:
            return gdn_tinv_sfpu_supported(chunk_size, any_input) ? GdnTinv::SFPU_FP32 : GdnTinv::HORNER;
    }
    TT_FATAL(false, "chunk_gdn: unknown wy_inverse {}", static_cast<uint32_t>(wy_inverse));
    return GdnTinv::HORNER;  // unreachable
}

inline void validate_gdn_tinv(GdnTinv tinv, uint32_t chunk_size, const Tensor& any_input) {
    if (tinv == GdnTinv::HORNER) {
        return;
    }
    TT_FATAL(
        chunk_size == tt::constants::TILE_HEIGHT,
        "chunk_gdn: the SFPU WY-inverse solve (wy_inverse=FORWARD_SUBSTITUTION) needs chunk_size == 32 (got {})",
        chunk_size);
    TT_FATAL(
        any_input.device()->arch() == tt::ARCH::BLACKHOLE,
        "chunk_gdn: the SFPU WY-inverse solve (wy_inverse=FORWARD_SUBSTITUTION) is Blackhole-only");
}

// Compile-time defines of the prep compute kernel: GDN_TINV_SFPU selects the solve; GDN_HOIST_RECONFIG is the
// fused producer's hoisted WY-path reconfigs (chunk_gdn_math.hpp, kGdnHoistReconfig).
inline tt::tt_metal::KernelDescriptor::Defines gdn_prep_defines(GdnTinv tinv, bool hoist_reconfig) {
    tt::tt_metal::KernelDescriptor::Defines defines;
    if (hoist_reconfig) {
        defines.emplace_back("GDN_HOIST_RECONFIG", "1");
    }
    if (tinv == GdnTinv::SFPU_FP32) {
        defines.emplace_back("GDN_TINV_SFPU", "1");
    }
    return defines;
}

}  // namespace ttnn::prim
