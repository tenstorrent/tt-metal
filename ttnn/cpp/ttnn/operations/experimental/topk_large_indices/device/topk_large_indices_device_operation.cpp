// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_device_operation.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

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
    return TensorSpec(
        ttnn::Shape(std::move(output_shape_vec)),
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
}

tensor_return_value_t TopkLargeIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<TopkLargeIndicesDeviceOperation::operation_attributes_t, TopkLargeIndicesDeviceOperation::tensor_args_t>
TopkLargeIndicesDeviceOperation::invoke(const Tensor& input_tensor, uint32_t k, std::optional<uint32_t> valid_length) {
    return {operation_attributes_t{.k = k, .valid_length = valid_length}, tensor_args_t(input_tensor)};
}

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

Tensor topk_large_indices(const Tensor& input_tensor, uint32_t k, std::optional<uint32_t> valid_length) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation::invoke(
            input_tensor, k, valid_length);
    return ttnn::device_operation::launch<
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation>(
        operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
