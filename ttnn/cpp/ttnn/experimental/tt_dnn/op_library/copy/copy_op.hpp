// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "ttnn/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

enum class CopyOpParallelizationStrategy {
    MULTI_CORE
};

struct Copy {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    CopyOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

operation::ProgramWithCallbacks copy_multi_core(const Tensor &input, const Tensor &output, bool backwards = false);

Tensor copy(const Tensor& src_tensor, const Tensor& dst_tensor);

Tensor clone(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype = std::nullopt);

Tensor typecast(const Tensor& input_tensor, const DataType& dtype, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

//unary assign
Tensor assign(const Tensor& input, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype = std::nullopt);

// binary assign
Tensor assign(const Tensor& input_a, const Tensor& input_b);

// binary assign with queue_id
Tensor assign(uint8_t queue_id, const Tensor& input_a, const Tensor& input_b);

}  // namespace tt_metal

}  // namespace tt
