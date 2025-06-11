// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include "padded_slice_op.hpp"
#include "padded_slice_program_factory.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental {

void PaddedSliceDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const bool has_step = std::any_of(this->step.cbegin(), this->step.cend(), [](uint32_t s) { return s != 1; });
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.padded_shape().rank() == 4, "Only 4D tensors are supported for padded_slice");
    TT_FATAL(
        input_tensor_a.padded_shape().rank() == this->padded_slice_start.rank() &&
            this->padded_slice_start.rank() == this->padded_slice_end.rank(),
        "Padded slice start, end and input tensor must all have the same rank");
    for (uint32_t i = 0; i < input_tensor_a.padded_shape().rank(); i++) {
        TT_FATAL(
            this->padded_slice_start[i] < input_tensor_a.padded_shape()[i],
            "Starts {} must be less than the shape of the tensor {} at index {}",
            this->padded_slice_start[i],
            input_tensor_a.padded_shape()[i],
            i);
        TT_FATAL(
            this->padded_slice_end[i] <= input_tensor_a.padded_shape()[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            this->padded_slice_end[i],
            input_tensor_a.padded_shape()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            this->padded_slice_start[i] <= this->padded_slice_end[i],
            "Slice start {} must be less than or equal to the end {}",
            this->padded_slice_start[i],
            this->padded_slice_end[i]);
    }
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        const auto output_shape_required = compute_output_specs(input_tensors)[0].logical_shape();
        const auto& out_tensor = output_tensors[0].value();
        TT_FATAL(
            out_tensor.padded_shape() == output_shape_required,
            "The input tensors need a shape of {}, however the output tensor is only {}",
            output_shape_required,
            out_tensor.padded_shape());
    }
    auto output_tensor_shape = this->compute_output_specs(input_tensors)[0].logical_shape();
    TT_FATAL(!has_step, "Padded slice does not support strided slices");
}

std::vector<ttnn::TensorSpec> PaddedSliceDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    SmallVector<uint32_t> out_shape(input_tensor.logical_shape().rank());

    TT_FATAL(out_shape.size() == 4, "Only 4D tensors are supported for padded_slice");
    auto output_dim_i = [this](size_t i) {
        return (this->padded_slice_end[i] - this->padded_slice_start[i] + this->step[i] - 1) / this->step[i];
    };
    for (uint32_t i = 0; i < out_shape.size(); i++) {
        out_shape[i] = output_dim_i(i);
    }
    out_shape[2] = out_shape[0] * out_shape[1] * out_shape[2];
    out_shape[0] = 1;
    out_shape[1] = 1;

    if (this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto output_shard_shape = this->output_mem_config.shard_spec().value().shape;
        out_shape[out_shape.size() - 1] = output_shard_shape[1];
    }

    ttnn::Shape output_tensor_shape(std::move(out_shape));
    auto output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();
    auto tensor_layout = TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), this->output_mem_config);
    return {ttnn::TensorSpec(output_tensor_shape, tensor_layout)};
}

operation::ProgramWithCallbacks PaddedSliceDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return detail::padded_slice_multi_core(
        input_tensor_a, output_tensor, this->padded_slice_start, this->padded_slice_end, this->step);
}

}  // namespace ttnn::operations::experimental
