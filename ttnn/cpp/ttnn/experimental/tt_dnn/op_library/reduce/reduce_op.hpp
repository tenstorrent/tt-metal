// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "ttnn/operation.hpp"

namespace tt {

namespace tt_metal {
enum class ReduceOpMath {
    SUM, MAX, MIN
};

enum class ReduceOpDim {
    H, W, HW
};

enum class ReduceOpParallelizationStrategy {
    MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE_HW
};

// TODO: Accept parallelization
operation::ProgramWithCallbacks reduce_single_core_hw(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath reduce_math, float scaler = 1.0f);
operation::ProgramWithCallbacks reduce_multi_core_h(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath reduce_math, float scaler = 1.0f);
operation::ProgramWithCallbacks reduce_multi_core_w(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath reduce_math, float scaler = 1.0f);

struct Reduce {
    const ReduceOpMath math_op;
    const ReduceOpDim dim;
    const float scaler;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    ReduceOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;
};

Tensor reduce(const Tensor &input_tensor, ReduceOpMath reduce_math, ReduceOpDim reduce_dim, float scaler = 1.0f, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const std::optional<DataType>& output_dtype=std::nullopt);
Tensor sum(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor max(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor min(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mean(const Tensor& input_tensor, uint aggregate_dims=2, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mean_hw(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor global_mean(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor global_sum(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor global_max(const Tensor& val, const MemoryConfig& output_mem_config);
Tensor global_min(const Tensor& val, const MemoryConfig& output_mem_config);

}  // namespace tt_metal

}  // namespace tt

namespace reduce_op_utils {

std::map<string, string> get_defines(ReduceOpMath reduce_op, ReduceOpDim reduce_dim);

} // namespace reduce_op_utils
