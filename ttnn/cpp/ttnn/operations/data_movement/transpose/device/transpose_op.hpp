// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

enum class TransposeOpDim { WH, HC, CN, NH, NW, CW };

enum class TransposeOpParallelizationStrategy { MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN };

struct Transpose {
    const TransposeOpDim dim;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<float> pad_value;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    TransposeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;
};

}  // namespace ttnn::operations::data_movement
