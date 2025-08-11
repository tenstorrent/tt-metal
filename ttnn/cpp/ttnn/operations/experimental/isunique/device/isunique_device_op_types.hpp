// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../isunique_common.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::operations::experimental::isunique {

using namespace common;
using namespace ttnn;
using namespace tt::tt_metal;

struct operation_attributes_t {
    const bool invert;
    const std::optional<int32_t> dim;
    const OptimalHeuristic optimal_heuristic;
    const std::optional<MemoryConfig> memory_config;
};

struct tensor_args_t {
    const Tensor input_tensor;
    const Tensor index_hint_tensor;
    const std::optional<Tensor> first_occurrences_tensor;
    const std::optional<Tensor> optional_out;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::isunique
