// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only: pin one implementation so tests and benchmarks can compare the two paths. Not for production use.

ttnn::Tensor concat_force_native(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    unsigned int groups,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Raises instead of falling back to native, so a comparison cannot measure native twice.
ttnn::Tensor concat_force_codegen(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    unsigned int groups,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::detail
