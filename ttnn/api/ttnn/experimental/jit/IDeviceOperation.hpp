// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include <vector>

namespace ttnn::experimental::jit {

struct IDeviceOperation {
    virtual void validate(const std::vector<Tensor>& input_tensors) const = 0;
    virtual std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const = 0;
    virtual tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const = 0;

    virtual Tensor invoke(std::vector<Tensor> input_tensors) = 0;
    virtual ~IDeviceOperation() = default;
};

}  // namespace ttnn::experimental::jit
