// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gridsample_op.hpp"
#include <algorithm>
#include <cmath>

#include <tt-metalium/util.hpp>
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::gridsample {
using namespace tt;
using namespace tt::tt_metal;

void gridsample::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& grid = input_tensors.at(1);

    TT_FATAL(
        input_tensor_a.get_logical_shape()[0] == grid.get_logical_shape()[0],
        "Input tensor and grid should have same batch size");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Input tensor layout should be ROW_MAJOR");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::FLOAT32, "Input tensor data type should be BFLOAT16");
    TT_FATAL(input_tensor_a.get_logical_shape().rank() == 4, "Input tensor should be in 4D");
    TT_FATAL(mode == "bilinear", "Upsample only supports bilinear or for now");
}

ttnn::Shape gridsample::compute_output_shape(const ttnn::Shape& input_shape, const ttnn::Shape& grid_shape) const {
    return ttnn::Shape{input_shape[0], input_shape[1], grid_shape[1], grid_shape[2]};
}

std::vector<TensorSpec> gridsample::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);

    const auto& grid = input_tensors.at(1);

    auto output_shape = compute_output_shape(input.get_logical_shape(), grid.get_logical_shape());

    return {
        TensorSpec(output_shape, TensorLayout(input.get_dtype(), PageConfig(input.get_layout()), output_mem_config_))};
}

operation::ProgramWithCallbacks gridsample::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const Tensor& input_tensor_0 = input_tensors.at(0);
    Tensor& output_tensor_0 = output_tensors.at(0);
    const Tensor& reshaped_input = input_tensors.at(2);

    return gridsample_rm_single_core(
        input_tensor_0, output_tensor_0, reshaped_input, normalized_grid, mode, align_corners);
}
}  // namespace ttnn::operations::gridsample
