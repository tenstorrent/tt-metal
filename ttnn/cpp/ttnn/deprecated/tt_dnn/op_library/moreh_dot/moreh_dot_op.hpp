/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

/*
 * dot product
 */
operation::ProgramWithCallbacks moreh_dot_single_core(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor);

struct MorehDot {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;  // TODO: Uplift output_dtype as an option for general dot/bmm

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

inline Tensor moreh_dot(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};

    operation::launch_op(
        [mem_config, input_tensor_a](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehDot{.output_mem_config = mem_config, .output_dtype = input_tensor_a.get_dtype()},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input_tensor_a, input_tensor_b},
        output_tensors);

    return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
