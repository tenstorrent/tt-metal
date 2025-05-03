// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::transformer {

enum class RotateHalfOpParallelizationStrategy { SINGLE_CORE };

struct RotateHalf {
    const tt::tt_metal::MemoryConfig output_mem_config;

    RotateHalfOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::transformer
