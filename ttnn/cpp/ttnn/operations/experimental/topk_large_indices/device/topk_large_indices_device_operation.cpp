// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_device_operation.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

#include <algorithm>
#include <limits>

namespace ttnn::operations::experimental::topk_large_indices {

namespace {

constexpr uint32_t max_supported_k = 2048;
constexpr uint32_t max_row_elements = 1u << 30;

void validate_static_args(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    // These fields are part of the program hash or process/device setup. They are validated on cache
    // miss before compiling the program and do not need to be rechecked on every cache hit.
    TT_FATAL(
        attrs.k > 0 && attrs.k <= max_supported_k && attrs.k % 16 == 0,
        "topk_large_indices supports k in [16, {}] in multiples of 16, got {}",
        max_supported_k,
        attrs.k);
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(
        arch == tt::ARCH::BLACKHOLE, "topk_large_indices is only supported on Blackhole architecture, got {}", arch);
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "topk_large_indices input must be ROW_MAJOR");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "topk_large_indices input must be BFLOAT16");
    TT_FATAL(!input.is_sharded(), "topk_large_indices input must use interleaved memory");
    if (tensor_args.input_indices.has_value()) {
        const auto& input_indices = tensor_args.input_indices.value();
        TT_FATAL(attrs.return_values, "topk_large_indices carried IDs require the value-preserving form");
        TT_FATAL(input_indices.layout() == Layout::ROW_MAJOR, "topk_large_indices carried IDs must be ROW_MAJOR");
        TT_FATAL(input_indices.dtype() == DataType::UINT32, "topk_large_indices carried IDs must be UINT32");
        TT_FATAL(!input_indices.is_sharded(), "topk_large_indices carried IDs must use interleaved memory");
        TT_FATAL(
            input_indices.memory_config() == input.memory_config(),
            "topk_large_indices values and carried IDs must use the same memory config");
    }
}

void validate_runtime_args(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    // Shape is intentionally omitted from the program hash and patched through runtime args, so keep
    // these checks on both cache miss and cache hit.
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "topk_large_indices input must be on device");
    TT_FATAL(input.buffer() != nullptr, "topk_large_indices input must have an allocated buffer");

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() >= 1, "topk_large_indices input must have rank >= 1");
    const uint64_t num_rows = flattened_rows_excluding_last_dim(shape);
    TT_FATAL(num_rows > 0, "topk_large_indices input must have at least one row");
    const uint32_t n = shape[shape.rank() - 1];
    TT_FATAL(n >= attrs.k, "topk_large_indices input last dimension {} must be >= k {}", n, attrs.k);
    TT_FATAL(
        n <= max_row_elements,
        "topk_large_indices initial implementation supports at most {} elements in the last dimension; got {}",
        max_row_elements,
        n);
    const uint64_t input_row_bytes = static_cast<uint64_t>(n) * input.element_size();
    TT_FATAL(
        input_row_bytes <= std::numeric_limits<uint32_t>::max(),
        "topk_large_indices input row size must fit in uint32_t bytes; got {} bytes",
        input_row_bytes);

    // Optional bounded search width: top-k scans only the first valid_length columns of each row (the rest
    // of the physically-wider row is ignored, not read). Must hold at least k values and fit within the row.
    if (attrs.valid_length.has_value()) {
        const uint32_t valid_length = attrs.valid_length.value();
        TT_FATAL(valid_length >= attrs.k, "topk_large_indices valid_length {} must be >= k {}", valid_length, attrs.k);
        TT_FATAL(
            valid_length <= n,
            "topk_large_indices valid_length {} must be <= the input last dimension {}",
            valid_length,
            n);
    }
    const uint64_t largest_index = static_cast<uint64_t>(attrs.index_offset) + attrs.valid_length.value_or(n) - 1;
    TT_FATAL(
        largest_index < std::numeric_limits<uint32_t>::max(),
        "topk_large_indices logical indices must not collide with the UINT32 sentinel; got largest index {}",
        largest_index);
    if (tensor_args.input_indices.has_value()) {
        const auto& input_indices = tensor_args.input_indices.value();
        TT_FATAL(
            input_indices.storage_type() == StorageType::DEVICE, "topk_large_indices carried IDs must be on device");
        TT_FATAL(input_indices.buffer() != nullptr, "topk_large_indices carried IDs must have an allocated buffer");
        TT_FATAL(
            input_indices.device() == input.device(),
            "topk_large_indices values and carried IDs must be on the same device");
        TT_FATAL(
            input_indices.logical_shape() == input.logical_shape(),
            "topk_large_indices carried IDs shape {} must match values shape {}",
            input_indices.logical_shape(),
            input.logical_shape());
        TT_FATAL(
            n <= 2 * attrs.k,
            "topk_large_indices carried-ID merge supports at most 2*k candidates; got width {} and k {}",
            n,
            attrs.k);
        TT_FATAL(
            !attrs.valid_length.has_value() || attrs.valid_length.value() == n,
            "topk_large_indices carried-ID merge must rank the complete 2*k candidate row");
        TT_FATAL(
            attrs.index_offset == 0,
            "topk_large_indices carried IDs and generated index offsets are mutually exclusive");
    }
}

}  // namespace

void TopkLargeIndicesDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_runtime_args(attrs, tensor_args);
}

void TopkLargeIndicesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_static_args(attrs, tensor_args);
    validate_runtime_args(attrs, tensor_args);
}

ttsl::hash::hash_t TopkLargeIndicesDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto grid = input.device()->compute_with_storage_grid_size();

    return tt::tt_metal::operation::hash_operation<TopkLargeIndicesDeviceOperation>(
        attrs.k,
        attrs.return_values,
        tensor_args.input_indices.has_value(),
        input.dtype(),
        input.layout(),
        input.memory_config().memory_layout(),
        input.memory_config().buffer_type(),
        grid.x,
        grid.y);
}

spec_return_value_t TopkLargeIndicesDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    std::vector<uint32_t> output_shape_vec;
    output_shape_vec.reserve(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        output_shape_vec.push_back(input_shape[i]);
    }
    output_shape_vec.back() = attrs.k;

    const auto memory_config = tensor_args.input_tensor.memory_config();
    const ttnn::Shape output_shape(output_shape_vec);
    if (!attrs.return_values) {
        output_shape_vec.back() = 0;
    }
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape(std::move(output_shape_vec)),
            tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config)),
        tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config))};
}

tensor_return_value_t TopkLargeIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto [values_spec, indices_spec] = compute_output_specs(attrs, tensor_args);
    return {
        create_device_tensor(values_spec, tensor_args.input_tensor.device()),
        create_device_tensor(indices_spec, tensor_args.input_tensor.device())};
}

std::tuple<TopkLargeIndicesDeviceOperation::operation_attributes_t, TopkLargeIndicesDeviceOperation::tensor_args_t>
TopkLargeIndicesDeviceOperation::invoke(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    uint32_t index_offset,
    const std::optional<Tensor>& input_indices,
    bool return_values) {
    return {
        operation_attributes_t{
            .k = k, .return_values = return_values, .index_offset = index_offset, .valid_length = valid_length},
        tensor_args_t{.input_tensor = input_tensor, .input_indices = input_indices}};
}

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

Tensor topk_large_indices(
    const Tensor& input_tensor, uint32_t k, std::optional<uint32_t> valid_length, uint32_t index_offset) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation::invoke(
            input_tensor,
            k,
            valid_length,
            index_offset,
            std::nullopt,
            /*return_values=*/false);
    auto [values, indices] =
        ttnn::device_operation::launch<operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation>(
            operation_attributes, tensor_args);
    values.deallocate();
    return indices;
}

std::tuple<Tensor, Tensor> topk_large_values_indices(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    uint32_t index_offset,
    const std::optional<Tensor>& input_indices) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation::invoke(
            input_tensor,
            k,
            valid_length,
            index_offset,
            input_indices,
            /*return_values=*/true);
    return ttnn::device_operation::launch<
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation>(
        operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
