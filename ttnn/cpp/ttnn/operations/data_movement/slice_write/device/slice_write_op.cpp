// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "slice_write_op.hpp"
#include "slice_write_program_factory.hpp"
#include "tt-metalium/assert.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

inline __attribute__((always_inline)) uint32_t get_upper_dims_compressed(const ttnn::Shape& shape) {
    return std::accumulate(shape.cbegin(), shape.cend() - 2, 1, std::multiplies<uint32_t>{});
}

inline __attribute__((always_inline)) uint32_t
get_upper_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start) {
    // offset for every dim except last 2
    uint32_t start_offset = 0;
    const auto& shape = tensor.get_padded_shape();

    uint32_t num_pages = tensor.volume();
    if (tensor.get_layout() == Layout::TILE) {
        num_pages /= tt::constants::TILE_HW;
    } else {
        uint32_t page_width = shape[-1];
        num_pages /= page_width;
    }

    for (uint32_t dim_outer = 0; dim_outer < shape.rank() - 2; dim_outer++) {
        uint32_t compressed_dims = 1;
        for (uint32_t dim_inner = 0; dim_inner <= dim_outer; dim_inner++) {
            compressed_dims *= shape[dim_inner];
        }
        start_offset += (num_pages / compressed_dims) * slice_start[dim_outer];
    }
    return start_offset;
}

void SliceWriteDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const bool has_step = std::any_of(this->step.cbegin(), this->step.cend(), [](uint32_t s) { return s != 1; });
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0).value();
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE || input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(
        input_tensor_a.get_padded_shape().rank() == this->slice_start.rank() &&
            this->slice_start.rank() == this->slice_end.rank(),
        "Error");
    for (uint32_t i = 0; i < input_tensor_a.get_padded_shape().rank(); i++) {
        TT_FATAL(this->slice_start[i] < input_tensor_a.get_padded_shape()[i], "Error");
        TT_FATAL(
            this->slice_end[i] <= input_tensor_a.get_padded_shape()[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            this->slice_end[i],
            input_tensor_a.get_padded_shape()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(this->slice_start[i] <= this->slice_end[i], "Error");
    }
    TT_FATAL(!output_tensors.empty(), "Output tensor is not provided to Slice Write.");
    auto output_tensor_shape = output_tensor.get_logical_shape();
    if (has_step) {  // if all ones modify before passing in to function
        TT_FATAL(
            input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Strided slice is only supported for row major layout");
        TT_FATAL(!input_tensor_a.is_sharded(), "Strided slice is not supported for sharded tensor");
        TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Strided slice is only supported for BFLOAT16");
        TT_FATAL(
            step.size() == this->slice_end.rank(),
            "Number of steps {} must match number of ends/starts {}",
            step.size(),
            this->slice_end.rank());
    }
    if (input_tensor_a.get_layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0, "Error");
        TT_FATAL(
            (output_tensor_shape[-2] % TILE_HEIGHT == 0) && (this->slice_start[-2] % TILE_HEIGHT == 0),
            "Can only unpad tilized tensor with full tiles");
        TT_FATAL(
            (output_tensor_shape[-1] % TILE_WIDTH == 0) && (this->slice_start[-1] % TILE_WIDTH == 0),
            "Can only unpad tilized tensor with full tiles");
    } else if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        if (has_step) {
            for (uint32_t i = 0; i < input_tensor_a.get_padded_shape().rank(); i++) {
                TT_FATAL(step[i] > 0, "Step({}) = {} should be positive", i, step[i]);
            }
        }
    }
}

std::vector<ttnn::Tensor> create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<ttnn::Tensor>>& output_tensors) {
    TT_FATAL(output_tensors.size() == 1, "A Single Output tensor should be provided to Slice Write.");
    TT_FATAL(output_tensors[0].has_value(), "Output tensor is not provided to Slice Write.");
    return {output_tensors[0].value()};
}

std::vector<ttnn::TensorSpec> compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<ttnn::Tensor>>& output_tensors) {
    TT_FATAL(output_tensors.size() == 1, "A Single Output tensor should be provided to Slice Write.");
    TT_FATAL(output_tensors[0].has_value(), "Output tensor is not provided to Slice Write.");
    return {output_tensors[0].value().get_tensor_spec()};
}

operation::ProgramWithCallbacks SliceWriteDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return detail::slice_write_multi_core(
        input_tensor_a, output_tensor, this->slice_start, this->slice_end, this->step);
}

}  // namespace ttnn::operations::data_movement
