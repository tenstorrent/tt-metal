// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "slice_op.hpp"
#include "slice_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

inline __attribute__((always_inline)) uint32_t get_upper_dims_compressed(const ttnn::Shape& shape) {
    return std::accumulate(shape.cbegin(), shape.cend() - 2, 1, std::multiplies<uint32_t>{});
}

inline __attribute__((always_inline)) uint32_t
get_upper_start_offset(const ttnn::Shape& shape, Layout layout, const ttnn::Shape& slice_start) {
    // offset for every dim except last 2
    uint32_t start_offset = 0;

    uint32_t num_pages = shape.volume();
    if (layout == Layout::TILE) {
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

inline __attribute__((always_inline)) uint32_t
get_upper_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start) {
    return get_upper_start_offset(tensor.padded_shape(), tensor.layout(), slice_start);
}

// Returns the start offset for a tiled tensor, given the input tensor and the slice start shape.
// If round_up is true, and the slice_start is not aligned to a tile boundary, it will round up to the next tile.
uint32_t get_tiled_start_offset(const ttnn::Shape& input_shape, const ttnn::Shape& slice_start, bool round_up) {
    using namespace tt::constants;
    uint32_t num_input_pages = input_shape.volume() / (TILE_HW);
    uint32_t upper_dims_compressed = get_upper_dims_compressed(input_shape);
    uint32_t num_pages_width = num_input_pages / (upper_dims_compressed * tt::div_up(input_shape[-2], TILE_HEIGHT));

    // offset for every dim except last 2
    uint32_t start_offset = get_upper_start_offset(input_shape, Layout::TILE, slice_start);

    if (round_up) {
        start_offset +=
            tt::div_up(slice_start[-2], TILE_HEIGHT) * num_pages_width + tt::div_up(slice_start[-1], TILE_WIDTH);
    } else {
        start_offset += slice_start[-2] / TILE_HEIGHT * num_pages_width + slice_start[-1] / TILE_WIDTH;
    }
    return start_offset;
}

uint32_t get_tiled_start_offset(const Tensor& input_tensor, const ttnn::Shape& slice_start, bool round_up) {
    const auto& shape = input_tensor.padded_shape();
    return get_tiled_start_offset(shape, slice_start, round_up);
}

uint32_t get_rm_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start) {
    uint32_t start_offset = 0;

    if (tensor.padded_shape().rank() >= 2) {
        const auto& shape = tensor.padded_shape();
        uint32_t num_pages = tensor.physical_volume() / shape[-1];
        uint32_t upper_dims_compressed = get_upper_dims_compressed(shape);
        start_offset = get_upper_start_offset(tensor, slice_start);
        start_offset += slice_start[-2];
    }

    return start_offset;
}

void SliceDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const bool has_step = std::any_of(this->step.cbegin(), this->step.cend(), [](uint32_t s) { return s != 1; });
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(
        input_tensor_a.padded_shape().rank() == this->slice_start.rank() &&
            this->slice_start.rank() == this->slice_end.rank(),
        "Error");
    for (uint32_t i = 0; i < input_tensor_a.padded_shape().rank(); i++) {
        TT_FATAL(this->slice_start[i] < input_tensor_a.padded_shape()[i], "Error");
        TT_FATAL(
            this->slice_end[i] <= input_tensor_a.padded_shape()[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            this->slice_end[i],
            input_tensor_a.padded_shape()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(this->slice_start[i] <= this->slice_end[i], "Error");
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
    if (has_step) {  // if all ones modify before passing in to function
        TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Strided slice is only supported for row major layout");
        TT_FATAL(!input_tensor_a.is_sharded(), "Strided slice is not supported for sharded tensor");
        TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Strided slice is only supported for BFLOAT16");
        TT_FATAL(
            step.size() == this->slice_end.rank(),
            "Number of steps {} must match number of ends/starts {}",
            step.size(),
            this->slice_end.rank());
    }
    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.physical_volume() % TILE_HW == 0, "Error");
        TT_FATAL(
            (output_tensor_shape[-2] % TILE_HEIGHT == 0) && (this->slice_start[-2] % TILE_HEIGHT == 0),
            "Can only slice tilized tensor with height begin index aligned to tiles");
        TT_FATAL(
            (output_tensor_shape[-1] % TILE_WIDTH == 0) && (this->slice_start[-1] % TILE_WIDTH == 0),
            "Can only slice tilized tensor with width begin index aligned to tiles");
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        if (has_step) {
            for (uint32_t i = 0; i < input_tensor_a.padded_shape().rank(); i++) {
                TT_FATAL(step[i] > 0, "Step({}) = {} should be positive", i, step[i]);
            }
        }
    }
}

std::vector<ttnn::TensorSpec> SliceDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    SmallVector<uint32_t> out_shape(input_tensor.logical_shape().rank());

    auto output_dim_i = [this](size_t i) {
        return (this->slice_end[i] - this->slice_start[i] + this->step[i] - 1) / this->step[i];
    };
    for (uint32_t i = 0; i < out_shape.size(); i++) {
        out_shape[i] = output_dim_i(i);
    }
    ttnn::Shape output_tensor_shape(std::move(out_shape));
    return {ttnn::TensorSpec(
        output_tensor_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), this->output_mem_config))};
}

operation::ProgramWithCallbacks SliceDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return detail::slice_multi_core(input_tensor_a, output_tensor, this->slice_start, this->slice_end, this->step);
}

}  // namespace ttnn::operations::data_movement
