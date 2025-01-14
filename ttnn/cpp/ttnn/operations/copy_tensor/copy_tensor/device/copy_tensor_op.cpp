// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_tensor_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include "ttnn/common/constants.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::copy_tensor {

void CopyTensor::validate(const std::vector<Tensor>& input_tensors) const {}

std::vector<ttnn::TensorSpec> CopyTensor::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    return {ttnn::TensorSpec(
        ttnn::Shape{32, 32},
        tt::tt_metal::TensorLayout(input_tensors[0].dtype(), PageConfig(input_tensors[0].layout()), MemoryConfig{}))};
}

// TODO: Remove output tensor entirely (if possible)
std::vector<ttnn::Shape> CopyTensor::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {ttnn::Shape{32, 32}};
}
std::vector<Tensor> CopyTensor::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_tensor = create_device_tensor(
        ttnn::Shape{32, 32},
        input_tensors[0].dtype(),
        input_tensors[0].layout(),
        input_tensors[0].device(),
        MemoryConfig{});
    std::vector<Tensor> output_tensors = {output_tensor};
    return output_tensors;
}
operation::ProgramWithCallbacks CopyTensor::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return copy_tensor_multi_core(input_tensors.at(0), input_tensors.at(1));
}

}  // namespace ttnn::operations::copy_tensor
