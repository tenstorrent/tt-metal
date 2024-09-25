// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "plusone_op.hpp"
#include "plusone_program_factory.hpp"

namespace ttnn::operations::experimental {

void PlusOne::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);

    TT_FATAL(input_tensor_a.get_dtype() == DataType::INT32, "Only INT32 is supported for inputs!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    auto input_shape = input_tensor_a.get_shape().with_tile_padding();
    TT_FATAL(input_shape.size()==1, "must have 1 dimension");

}

std::vector<ttnn::Shape> PlusOne::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors[0].get_shape().with_tile_padding();
    return {input_shape};
}

std::vector<Tensor> PlusOne::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    return {input_tensors.at(0)};

}

operation::ProgramWithCallbacks PlusOne::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    return detail::plusone_single_core(input_tensor);
}

}  // namespace ttnn::operations::experimental
