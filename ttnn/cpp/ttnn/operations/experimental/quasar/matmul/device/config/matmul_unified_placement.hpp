// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

namespace ttnn::operations::experimental::quasar::matmul {

// A config is what the caller asks for, its placement is the cores, C slice and orientation, and the plan
// (UnifiedMatmulPlan) is everything derived from a config with a placement.
// Returns `requested` with the placement whose plan has the lowest estimated cycles among the natural ones; nullopt
// when the unified matmul cannot run the case or no placement fits.
std::optional<MatmulUnifiedProgramConfig> choose_unified_placement(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    uint32_t bias_single_tile_size,
    const ttnn::prim::qsr::MatmulParams& attributes,
    const MatmulUnifiedProgramConfig& requested);

}  // namespace ttnn::operations::experimental::quasar::matmul
