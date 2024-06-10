// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_op.hpp"
#include "argmax_program_factory.hpp"

namespace ttnn::operations::reduction {

void ArgMax::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(this->output_dtype == DataType::UINT32, "Only UINT32 is supported for outputs!");
    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for outputs!");

    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    const auto &optional_output_tensor = output_tensors.at(0);
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().get_dtype() == DataType::UINT32, "Only UINT32 is supported for outputs!");
        TT_FATAL(
            optional_output_tensor.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }

    if (this->dim.has_value()) {
        const uint32_t input_rank = input_tensor_a.get_legacy_shape().rank();
        const uint32_t normalized_dim = dim.value() < 0 ? dim.value() + input_rank : dim.value();
        TT_FATAL(normalized_dim >= 0, fmt::format("Invalid dim for argmax: {}!", dim.value()));
        TT_FATAL(normalized_dim < input_rank, fmt::format("Invalid dim for argmax: {}!", dim.value()));

        // TODO: Add support for normalized_dim = 0, 1, 2
        TT_FATAL(normalized_dim == (input_rank - 1), "Only argmax on last dim is supported!");
    }
}

std::vector<tt::tt_metal::Shape> ArgMax::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors[0].get_legacy_shape();
    if (this->dim.has_value()) {
        // TODO: There seems to be an underflow issue with directly modifying last two dims
        if (this->dim.value() == -1 or this->dim.value() == 3) {
            tt::tt_metal::Shape output_shape({input_shape[0], input_shape[1], input_shape[2], 1});
            return {output_shape};
        } else if (this->dim.value() == -2 or this->dim.value() == 2) {
            tt::tt_metal::Shape output_shape({input_shape[0], input_shape[1], 1, input_shape[3]});
            return {output_shape};
        } else {
            input_shape[this->dim.value()] = 1;
            return {input_shape};
        }
    } else {
        tt::tt_metal::Shape output_shape({1, 1, 1, 1});
        return {output_shape};
    }
}

std::vector<Tensor> ArgMax::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    const auto &input_tensor = input_tensors[0];
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks ArgMax::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    const auto &output_tensor = output_tensors.at(0);

    return detail::argmax_multi_core(input_tensor, output_tensor, this->dim);
}

}  // namespace ttnn::operations::reduction
