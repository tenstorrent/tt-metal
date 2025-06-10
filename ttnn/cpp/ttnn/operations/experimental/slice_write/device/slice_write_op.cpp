// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "slice_write_op.hpp"
#include "slice_write_program_factory.hpp"
#include "tt-metalium/assert.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental {

void SliceWriteDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const bool has_step = std::any_of(this->step.cbegin(), this->step.cend(), [](uint32_t s) { return s != 1; });
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0).value();
    const auto output_padded_shape = output_tensor.padded_shape();
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(
        input_tensor_a.padded_shape().rank() == this->slice_start.rank() &&
            output_padded_shape.rank() == this->slice_start.rank() &&
            this->slice_start.rank() == this->slice_end.rank(),
        "Ranks of input tensor, output_tensor, slice start and slice end should be equal. Got {} {} {} {}",
        input_tensor_a.padded_shape().rank(),
        output_padded_shape.rank(),
        this->slice_start.rank(),
        this->slice_end.rank());
    for (uint32_t i = 0; i < output_padded_shape.rank(); i++) {
        TT_FATAL(
            this->slice_start[i] < output_padded_shape[i],
            "Start is outside the bounds of the output tensor for index {}. Got {}. Size {}",
            i,
            this->slice_start[i],
            output_padded_shape[i]);
        TT_FATAL(
            this->slice_end[i] <= output_padded_shape[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            this->slice_end[i],
            output_padded_shape[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            this->slice_start[i] <= this->slice_end[i],
            "Slice start {} should be less than slice end {}",
            this->slice_start[i],
            this->slice_end[i]);
    }
    TT_FATAL(
        this->slice_start[-1] == 0,
        "Slice write doesn't support slicing along the last dimension. Slice start [-1] should be 0");
    TT_FATAL(
        this->slice_end[-1] == output_padded_shape[-1],
        "Slice write doesn't support slicing along the last dimension. Slice end [-1] should be equal to output shape "
        "[-1]");
}

std::vector<ttnn::Tensor> SliceWriteDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<ttnn::Tensor>& output_tensors) const {
    TT_FATAL(output_tensors.size() == 1, "A Single Output tensor should be provided to Slice Write.");
    return output_tensors;
}

std::vector<ttnn::TensorSpec> SliceWriteDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    return {input_tensors[0].tensor_spec()};
}

operation::ProgramWithCallbacks SliceWriteDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return detail::slice_write_multi_core(
        input_tensor_a, output_tensor, this->slice_start, this->slice_end, this->step);
}

}  // namespace ttnn::operations::experimental
