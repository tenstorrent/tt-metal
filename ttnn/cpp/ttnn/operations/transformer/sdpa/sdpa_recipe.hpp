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
tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy, const CoreRangeSet& grid, uint32_t k_chunks);

Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config);

}  // namespace ttnn::operations::transformer::sdpa::detail
