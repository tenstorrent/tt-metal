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
    TT_FATAL(
        input.layout() == Layout::ROW_MAJOR || input.layout() == Layout::TILE,
        "topk_large_indices input must be ROW_MAJOR or TILE");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "topk_large_indices input must be BFLOAT16");
    TT_FATAL(!input.is_sharded(), "topk_large_indices input must use interleaved memory");
    if (attrs.tile_output) {
        TT_FATAL(
            attrs.k % 32 == 0,
            "topk_large_indices tile_output requires k to be a multiple of 32 (no partial output tile "
            "columns), got {}",
            attrs.k);
    }
    if (attrs.index_dtype.has_value()) {
        TT_FATAL(
            attrs.index_dtype == DataType::UINT32 || attrs.index_dtype == DataType::UINT16,
            "topk_large_indices index_dtype must be UINT32 or UINT16, got {}",
            attrs.index_dtype.value());
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
    // of the physically-wider row is ignored, not read).  A short valid prefix is legal: inactive lanes
    // are already represented as -inf by the TopK XL tail path and are materialized as 0xFFFFFFFF indices.
    // This keeps a fixed worst-case k/output shape while a serving cache grows through its early prefixes.
    if (attrs.valid_length.has_value()) {
        const uint32_t valid_length = attrs.valid_length.value();
        TT_FATAL(valid_length > 0, "topk_large_indices valid_length must be > 0");
        // NOTE: valid_length < k is supported by design — lanes beyond the prefix's
        // capacity emit the 0xFFFFFFFF sentinel index (and -inf values when
        // return_values is set). The public docstring's "[k, last dimension]" domain
        // is stale; the tested contract is (0, last dimension].
        TT_FATAL(
            valid_length <= n,
            "topk_large_indices valid_length {} must be <= the input last dimension {}",
            valid_length,
            n);
    }

    // UINT16 index emission: every real winner index must provably fit 16 bits. Winners are
    // positions < the searched width, so search_len <= 65535 guarantees winners <= 65534 and keeps
    // 0xFFFF unambiguous as the truncated -inf sentinel. Checked at runtime because the shape (and
    // valid_length) is runtime-patched while index_dtype is baked into the program.
    if (attrs.index_dtype == DataType::UINT16) {
        const uint32_t search_len = attrs.valid_length.value_or(n);
        TT_FATAL(
            search_len <= 65535,
            "topk_large_indices index_dtype=UINT16 requires the searched width (valid_length if set, else the "
            "last dimension) to be <= 65535 so winner indices fit 16 bits; got {}",
            search_len);
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

namespace {

program::ColumnSplitConfig column_split_config_for(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    const uint32_t num_rows = flattened_rows_excluding_last_dim(shape);
    const auto grid = input.device()->compute_with_storage_grid_size();
    return program::compute_column_split_config(attrs.k, n, num_rows, grid, attrs.num_slices);
}

}  // namespace

TopkLargeIndicesDeviceOperation::program_factory_t TopkLargeIndicesDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (column_split_config_for(attrs, tensor_args).enabled) {
        return program::TopkLargeIndicesMultiCoreProgramFactory{};
    }
    return program::TopkLargeIndicesProgramFactory{};
}

ttsl::hash::hash_t TopkLargeIndicesDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto grid = input.device()->compute_with_storage_grid_size();

    // Factory selection (and, on the column-parallel path, the program
    // structure) depends on the derived split config, so its fields must be
    // hashed. On the row-parallel path the config is all zeros, preserving the
    // original shape-free hash: row counts and row widths keep patching
    // through runtime args without recompiles.
    const auto split_config = column_split_config_for(attrs, tensor_args);

    return tt::tt_metal::operation::hash_operation<TopkLargeIndicesDeviceOperation>(
        attrs.k,
        // Selects the with-values kernels, extra CBs, and the second output.
        attrs.return_values,
        // Selects the tile-scatter writer kernels, output CBs, and output specs.
        attrs.tile_output,
        // Selects the narrowing writer path and the indices output spec.
        attrs.index_dtype,
        // User P override: also reflected in the derived split_config fields
        // below, but hashed directly so intent and derivation can never skew.
        attrs.num_slices,
        input.dtype(),
        input.layout(),
        input.memory_config().memory_layout(),
        input.memory_config().buffer_type(),
        grid.x,
        grid.y,
        split_config.enabled,
        split_config.num_slices,
        split_config.local_grid_x,
        split_config.local_grid_y);
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
    const ttnn::Shape output_shape(std::move(output_shape_vec));

    const auto memory_config = tensor_args.input_tensor.memory_config();
    const Layout output_layout = attrs.tile_output ? Layout::TILE : Layout::ROW_MAJOR;
    const DataType indices_dtype = attrs.index_dtype.value_or(DataType::UINT32);
    spec_return_value_t specs;
    specs.emplace_back(
        output_shape,
        tt::tt_metal::TensorLayout(indices_dtype, tt::tt_metal::PageConfig(output_layout), memory_config));
    if (attrs.return_values) {
        specs.emplace_back(
            output_shape,
            tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(output_layout), memory_config));
    }
    return specs;
}

tensor_return_value_t TopkLargeIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(attrs, tensor_args);
    tensor_return_value_t outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(create_device_tensor(spec, tensor_args.input_tensor.device()));
    }
    return outputs;
}

std::tuple<TopkLargeIndicesDeviceOperation::operation_attributes_t, TopkLargeIndicesDeviceOperation::tensor_args_t>
TopkLargeIndicesDeviceOperation::invoke(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    bool return_values,
    std::optional<uint32_t> num_slices,
    bool tile_output,
    std::optional<DataType> index_dtype) {
    return {
        operation_attributes_t{
            .k = k,
            .valid_length = valid_length,
            .return_values = return_values,
            .num_slices = num_slices,
            .tile_output = tile_output,
            .index_dtype = index_dtype},
        tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

Tensor topk_large_indices(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    std::optional<uint32_t> num_slices,
    bool tile_output,
    std::optional<DataType> index_dtype) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation::invoke(
            input_tensor, k, valid_length, /*return_values=*/false, num_slices, tile_output, index_dtype);
    auto outputs =
        ttnn::device_operation::launch<operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation>(
            operation_attributes, tensor_args);
    return std::move(outputs[0]);
}

std::tuple<Tensor, Tensor> topk_large_indices_with_values(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    std::optional<uint32_t> num_slices,
    bool tile_output,
    std::optional<DataType> index_dtype) {
    auto [operation_attributes, tensor_args] =
        operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation::invoke(
            input_tensor, k, valid_length, /*return_values=*/true, num_slices, tile_output, index_dtype);
    auto outputs =
        ttnn::device_operation::launch<operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation>(
            operation_attributes, tensor_args);
    // torch convention: (values, indices).
    return {std::move(outputs[1]), std::move(outputs[0])};
}

}  // namespace ttnn::experimental
