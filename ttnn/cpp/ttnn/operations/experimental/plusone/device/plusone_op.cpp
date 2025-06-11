// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "plusone_op.hpp"
#include "plusone_program_factory.hpp"

namespace ttnn::operations::experimental {

void PlusOne::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);

    TT_FATAL(
        input_tensor_a.dtype() == tt::tt_metal::DataType::INT32 ||
            input_tensor_a.dtype() == tt::tt_metal::DataType::UINT32,
        "Only INT32 and UINT32 is supported for inputs!");
    TT_FATAL(
        input_tensor_a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    auto input_shape = input_tensor_a.padded_shape();
    TT_FATAL(input_shape.size() >= 1 && input_shape.size() <= 4, "must have 1 to 4 dimensions for input tensor");
}

std::vector<ttnn::TensorSpec> PlusOne::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).tensor_spec()};
}

std::vector<Tensor> PlusOne::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    return {input_tensors.at(0)};
}

tt::tt_metal::operation::ProgramWithCallbacks PlusOne::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return detail::plusone_single_core(input_tensor, sub_core_grids);
}

}  // namespace ttnn::operations::experimental
