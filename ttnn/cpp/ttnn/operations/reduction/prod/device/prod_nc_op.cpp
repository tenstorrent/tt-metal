// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "prod_nc_op.hpp"

#include <tt-metalium/constants.hpp>

namespace tt {

using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Prod
////////////////////////////////////////////////////////////////////////////
void Prod::validate(const std::vector<Tensor>& inputs) const {
    TT_FATAL((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    const auto& input = inputs.at(0);
    const auto& output = inputs.at(1);

    auto input_shape = input.get_padded_shape();
    TT_FATAL((input_shape.rank() == 4), "rank should be 4");
    const auto& output_shape = output.get_padded_shape();
    auto input_shape_wo_padding = input.get_logical_shape();
    const auto& output_shape_wo_padding = output.get_logical_shape();

    if (dim == 0 || dim == 1) {
        input_shape[dim] = 1;
        input_shape_wo_padding[dim] = 1;
    }

    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_FATAL(input_shape[i] == output_shape[i], "Error");
        // TT_FATAL(input_shape_wo_padding[i] == output_shape_wo_padding[i], "Error");
    }
}

std::vector<Tensor> Prod::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<TensorSpec> Prod::compute_output_specs(const std::vector<Tensor>&) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks Prod::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& input = inputs.at(0);
    auto& output = inputs.at(1);

    return prod_nc_format(input, output, dim);
}

ttnn::Shape compute_output_shape(const ttnn::Shape& input_shape, const int64_t& dim) {
    auto output_shape = input_shape;
    switch (dim) {
        case 0:
        case 1: output_shape[dim] = 1; break;
    }

    return output_shape;
}

inline Tensor create_output_tensor(
    const Tensor& input_tensor, const ttnn::Shape& output_shape, const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);
    return create_device_tensor(
        output_shape, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config);
}

// output as arg
Tensor prod_(const Tensor& input, const Tensor& output, const int64_t& dim) {
    operation::run(Prod{.dim = dim}, {input, output});
    return output;
}

// output creation inside
Tensor prod_(const Tensor& input, const int64_t& dim, const MemoryConfig& mem_config) {
    const auto& input_shape = input.get_padded_shape();
    auto output_shape = compute_output_shape(input_shape, dim);
    auto output = create_output_tensor(input, output_shape, mem_config);

    operation::run(Prod{.dim = dim}, {input, output});
    return output;
}

Tensor prod_nc(
    const Tensor& input,
    const Tensor& output,
    ttnn::SmallVector<int64_t>& dims,
    const MemoryConfig& output_mem_config) {
    TT_FATAL(!dims.empty(), "prod_nc dims should not be empty");

    ttnn::SmallVector<int64_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(LogTest, "{}:{} dim {}", __func__, __LINE__, sorted_dims[i]);
        auto temp_output = prod_(temp_input, sorted_dims[i], output_mem_config);
        temp_input = temp_output;
    }
    log_debug(LogTest, "{}:{} dim {}", __func__, __LINE__, sorted_dims.front());
    prod_(temp_input, output, sorted_dims.front());
    return output;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
