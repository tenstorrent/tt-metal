// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_xl_device_operation.hpp"

#include <tt-metalium/math.hpp>

namespace ttnn::operations::experimental::topk_xl {

void TopkXLDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void TopkXLDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    TT_FATAL(
        attrs.k == 512 || attrs.k == 1024 || attrs.k == 2048,
        "topk_xl initial implementation supports k in {{512, 1024, 2048}}, got {}",
        attrs.k);
    TT_FATAL(attrs.largest, "topk_xl initial implementation supports largest=true only");
    TT_FATAL(attrs.sorted, "topk_xl initial implementation supports sorted=true only");

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "topk_xl input must be on device");
    TT_FATAL(input.buffer() != nullptr, "topk_xl input must have an allocated buffer");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "topk_xl input must be ROW_MAJOR");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "topk_xl input must be BFLOAT16");

    const auto shape = input.logical_shape();
    TT_FATAL(shape.rank() >= 1, "topk_xl input must have rank >= 1");
    const uint32_t n = shape[shape.rank() - 1];
    TT_FATAL(n >= attrs.k, "topk_xl input last dimension {} must be >= k {}", n, attrs.k);
    TT_FATAL(n % attrs.k == 0, "topk_xl initial implementation requires N % k == 0; got N={}, k={}", n, attrs.k);
    constexpr uint32_t max_row_elements = 131072;
    TT_FATAL(
        n <= max_row_elements,
        "topk_xl initial implementation supports at most {} elements in the last dimension; got {}",
        max_row_elements,
        n);
}

spec_return_value_t TopkXLDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto input_shape = tensor_args.input_tensor.logical_shape();
    std::vector<uint32_t> output_shape_vec;
    output_shape_vec.reserve(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        output_shape_vec.push_back(input_shape[i]);
    }
    output_shape_vec.back() = attrs.k;

    const auto memory_config = tensor_args.input_tensor.memory_config();
    auto values_spec = TensorSpec(
        ttnn::Shape(output_shape_vec),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
    auto indices_spec = TensorSpec(
        ttnn::Shape(std::move(output_shape_vec)),
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));

    return {values_spec, indices_spec};
}

tensor_return_value_t TopkXLDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    auto values = create_device_tensor(std::get<0>(specs), device);
    auto indices = create_device_tensor(std::get<1>(specs), device);
    return {values, indices};
}

std::tuple<TopkXLDeviceOperation::operation_attributes_t, TopkXLDeviceOperation::tensor_args_t>
TopkXLDeviceOperation::invoke(const Tensor& input_tensor, uint32_t k, bool largest, bool sorted) {
    return {
        operation_attributes_t{.k = k, .largest = largest, .sorted = sorted},
        tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::experimental::topk_xl

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> topk_xl(const Tensor& input_tensor, uint32_t k, bool largest, bool sorted) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_xl::TopkXLDeviceOperation::invoke(input_tensor, k, largest, sorted);
    return ttnn::device_operation::launch<operations::experimental::topk_xl::TopkXLDeviceOperation>(
        operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
