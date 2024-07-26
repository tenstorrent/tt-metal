// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::ssm {

struct SumReduce {
    MemoryConfig output_mem_config;
    DataType output_dtype;
    MathFidelity math_fidelity;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

inline Tensor sum_reduce(
    const Tensor& input_tensor_a,
    const MemoryConfig& mem_config,
    std::optional<const DataType> output_dtype = std::nullopt,
    MathFidelity math_fidelity = MathFidelity::HiFi4) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a}))};
    operation::launch_op(
        [mem_config, output_dtype, math_fidelity](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor_a = input_tensors.at(0);
            return operation::run(
                SumReduce{mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), math_fidelity}, input_tensors);
        },
        {input_tensor_a},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ttnn::operations::experimental::ssm
