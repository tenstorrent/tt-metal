// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

enum class ConcatOpParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct ConcatDeviceOperation {
    uint32_t dim;
    unsigned int groups;
    const tt::tt_metal::MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    ConcatOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;
};

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
// Notes: Non-empty tensors provided must have the same shape, except in the cat dimension.
Tensor concat_impl(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim = 0,
    unsigned int groups = 1,
    const tt::tt_metal::MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace ttnn::operations::data_movement
