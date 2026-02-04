// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "prod_nc_op.hpp"
#include "prod_nc_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace tt::operations::primary {

namespace {

ttnn::Shape compute_output_shape(const ttnn::Shape& input_shape, int64_t dim) {
    auto output_shape = input_shape;
    switch (dim) {
        case 0:
        case 1: output_shape[dim] = 1; break;
        default: TT_THROW("Unsupported dim {} for prod nc op", dim);
    }
    return output_shape;
}

Tensor create_output_tensor(
    const Tensor& input_tensor, const ttnn::Shape& output_shape, const MemoryConfig& mem_config) {
    TT_FATAL(
        input_tensor.storage_type() == tt_metal::StorageType::DEVICE,
        "Input tensor must be stored on device. Storage type: {}",
        input_tensor.storage_type());
    return create_device_tensor(
        ttnn::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config)),
        input_tensor.device());
}

// output as arg
Tensor prod_(const Tensor& input, const Tensor& output, const int64_t& dim) {
    ttnn::prim::prod_nc(input, output, dim);
    return output;
}

// output creation inside
Tensor prod_(const Tensor& input, const int64_t& dim, const MemoryConfig& mem_config) {
    const auto& input_shape = input.padded_shape();
    auto output_shape = compute_output_shape(input_shape, dim);
    auto output = create_output_tensor(input, output_shape, mem_config);

    ttnn::prim::prod_nc(input, output, dim);
    return output;
}

}  // namespace

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

}  // namespace tt::operations::primary
