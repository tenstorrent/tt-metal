// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

namespace ttnn::operations::experimental::quasar::matmul {

// Estimated device cycles of a plan: the largest of the busiest core's compute and data movement and the device's
// DRAM traffic, part of the other two (the buffering overlaps them imperfectly), and a fixed cost per C slice.
double estimate_unified_matmul_cycles(
    const ttnn::prim::qsr::UnifiedMatmulPlan& plan,
    tt::ARCH arch,
    tt::tt_metal::MathFidelity math_fidelity,
    bool A_in_dram,
    bool B_in_dram,
    bool C_in_dram);

// The placement (cores, C slice, orientation) with the lowest estimate among the natural ones, the other fields from
// `requested`; nullopt when the unified matmul cannot run the case or no placement fits.
std::optional<MatmulUnifiedProgramConfig> choose_unified_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    uint32_t bias_single_tile_size,
    const ttnn::prim::qsr::MatmulParams& attributes,
    const MatmulUnifiedProgramConfig& requested);

}  // namespace ttnn::operations::experimental::quasar::matmul
