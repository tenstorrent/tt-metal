// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "sdpa_precision_policy.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type);
// Q tiles per chunk for a validated recipe program config (defaults to Q256).
uint32_t recipe_q_tiles(const std::optional<SDPAProgramConfig>& program_config);

// K tiles per chunk for a validated recipe program config (defaults to K512).
uint32_t recipe_k_tiles(const std::optional<SDPAProgramConfig>& program_config);

tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles = 8,
    uint32_t k_tiles = 16,
    uint32_t d_tiles = 4);

PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    bool inputs_prepared,
    std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<SDPAProgramConfig>& program_config);

std::tuple<Tensor, Tensor> run_joint_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& joint_q,
    const Tensor& joint_k,
    const Tensor& joint_v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config);

Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config);

}  // namespace ttnn::operations::transformer::sdpa::detail
