// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fill_pad/device/fill_pad_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/fill_pad/device/fill_pad_program_factory.hpp"

namespace ttnn::operations::data_movement {

using namespace tt;
using namespace tt::tt_metal;

void FillPad::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.layout() == TILE_LAYOUT, "FillPad should only be used for tile layout");
    TT_FATAL(detail::data_type_to_size.count(input_tensor_a.dtype()), "Unsupported datatype");
}

std::vector<TensorSpec> FillPad::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.tensor_spec()};
}

std::vector<Tensor> FillPad::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor};
}

operation::ProgramWithCallbacks FillPad::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::fill_pad_multi_core(input_tensor, this->fill_value);
}

}  // namespace ttnn::operations::data_movement
