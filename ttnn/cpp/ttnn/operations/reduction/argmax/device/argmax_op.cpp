// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_op.hpp"
#include "argmax_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction {

ttnn::SmallVector<uint32_t> ArgMax::get_output_shape(const Tensor& input_tensor) const {
    auto input_shape = input_tensor.logical_shape();
    int rank = input_shape.size();
    ttnn::SmallVector<uint32_t> output_shape;

    // If no reduction dims are specified, we reduce all dimensions
    auto all_dim_reduce = not this->dim.has_value();
    auto red_dim = this->dim.value_or(0);
    TT_FATAL(
        (rank == 0) or ((red_dim >= -rank) and (red_dim < rank)),
        "Invalid reduction dimension {} for input tensor with rank {}",
        red_dim,
        rank);

    // Adjust negative reduction dimension to positive
    red_dim = red_dim < 0 ? red_dim + rank : red_dim;

    // Generate output shape
    // Iterate over the input shape and adjust the output shape for keepdim
    for (int dim = 0; dim < rank; ++dim) {
        // If this is in the reduction dims, keep it only if keepdim is true
        bool is_reduction_dim = all_dim_reduce or (dim == red_dim);

        if (is_reduction_dim) {
            TT_FATAL(input_shape[dim] != 0, "Expected reduction dim {} to have non-zero size", dim);
            if (this->keepdim) {
                output_shape.push_back(1);
            }
        } else {
            // If this is not a reduction dim, we keep the original size
            output_shape.push_back(input_shape[dim]);
        }
    }

    return output_shape;
}

void ArgMax::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    TT_FATAL(this->output_dtype == DataType::UINT32, "Only UINT32 is supported for outputs!");
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for outputs!");

    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    const auto& optional_output_tensor = output_tensors.at(0);
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor.value().dtype() == DataType::UINT32, "Only UINT32 is supported for outputs!");
        TT_FATAL(
            optional_output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }

    if (this->dim.has_value()) {
        const uint32_t input_rank = input_tensor_a.padded_shape().rank();
        const uint32_t normalized_dim = dim.value() < 0 ? dim.value() + input_rank : dim.value();

        // TODO: Add support for normalized_dim = 0, 1, 2
        TT_FATAL(normalized_dim == (input_rank - 1), "Only argmax on last dim is supported!");
    }

    if (this->use_multicore && this->sub_core_grids.has_value()) {
        TT_FATAL(
            this->sub_core_grids->ranges().size() <= 2,
            "Multicore argmax only supports up to 2 core grid ranges, but got {} ranges",
            this->sub_core_grids->ranges().size());
    }
}

std::vector<TensorSpec> ArgMax::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_tensor = input_tensors[0];
    auto output_shape = this->get_output_shape(input_tensor);
    return {TensorSpec(
        ttnn::Shape(output_shape), TensorLayout(output_dtype, PageConfig(input_tensor.layout()), output_mem_config))};
}

std::vector<Tensor> ArgMax::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors[0].device())};
}

operation::ProgramWithCallbacks ArgMax::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    const auto normalized_dim =
        this->dim.has_value() ? *this->dim + input_tensor.padded_shape().rank() * (*this->dim < 0) : this->dim;
    if (this->use_multicore) {
        return detail::argmax_multi_core(
            input_tensor, output_tensor, normalized_dim, this->keepdim, this->sub_core_grids);
    }
    return detail::argmax_single_core(input_tensor, output_tensor, normalized_dim, this->keepdim);
}

}  // namespace ttnn::operations::reduction
