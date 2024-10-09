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
 * dot product backward
 */
operation::ProgramWithCallbacks moreh_dot_backward_single_core(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    const std::optional<const Tensor> &input_grad,
    const std::optional<const Tensor> &other_grad);

struct MorehDotBackward {
    void validate(
        const std::vector<Tensor> &inputs, const std::vector<std::optional<const Tensor>> &optional_inputs) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs,
        const std::vector<std::optional<const Tensor>> &optional_inputs,
        std::vector<Tensor> &outputs) const;
};

std::vector<std::optional<Tensor>> moreh_dot_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    std::optional<const Tensor> input_grad = std::nullopt,
    std::optional<const Tensor> other_grad = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
