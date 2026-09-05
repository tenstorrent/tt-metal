// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_device_operation.hpp"

#include "ttnn/operations/data_movement/concat/concat.hpp"

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
    // of the physically-wider row is ignored, not read).  A short valid prefix is legal: inactive lanes
    // are already represented as -inf by the TopK XL tail path and are materialized as 0xFFFFFFFF indices.
    // This keeps a fixed worst-case k/output shape while a serving cache grows through its early prefixes.
    if (attrs.valid_length.has_value()) {
        const uint32_t valid_length = attrs.valid_length.value();
        TT_FATAL(valid_length > 0, "topk_large_indices valid_length must be > 0");
        // valid_length < k is supported by design — lanes beyond the prefix's
        // capacity emit the 0xFFFFFFFF sentinel index; the documented and
        // tested domain is (0, last dimension].
        TT_FATAL(
            valid_length <= n,
            "topk_large_indices valid_length {} must be <= the input last dimension {}",
            valid_length,
            n);
    }

    // Composite-internal row window: must map onto the canonical rows dimension (all leading dims 1)
    // and stay in bounds. Both fields travel together.
    TT_FATAL(
        attrs.row_start.has_value() == attrs.row_count.has_value(),
        "topk_large_indices row_start and row_count must be set together");
    if (attrs.row_count.has_value()) {
        TT_FATAL(
            shape.rank() >= 2 && num_rows == shape[shape.rank() - 2],
            "topk_large_indices row windows require the canonical [1.., R, W] shape (leading dims 1)");
        const uint64_t row_end = static_cast<uint64_t>(*attrs.row_start) + *attrs.row_count;
        TT_FATAL(
            *attrs.row_count > 0 && row_end <= num_rows,
            "topk_large_indices row window [{}, {}) out of bounds for {} rows",
            *attrs.row_start,
            row_end,
            num_rows);
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
    const uint32_t num_rows = attrs.row_count.value_or(flattened_rows_excluding_last_dim(shape));
    const auto grid = input.device()->compute_with_storage_grid_size();
    // The cost model may auto-select the multi-row rectangle engine when it
    // models a win: 2*ceil(chunks/P) + ceil(log2 P) beating the row-parallel
    // 2*chunks by the extra multi-row margin (measured routed 477 -> 330 us
    // at 32x65536 k=2048). Engine selection changes only WHICH of the tied
    // bf16 values win — tie identity is the documented non-stable contract;
    // the selected value multiset is engine-invariant. The program hash
    // carries the derived split-config fields, so an engine change
    // recompiles — bounded, because the model quantizes P to a handful of
    // choices per (k, width, rows, grid) and fixed-shape callers compile
    // once. Explicit (internal) num_slices still bypasses the model and pins
    // P directly (the hybrid wrapper's remainder window).
    return program::compute_column_split_config(attrs.k, n, num_rows, grid, attrs.num_slices, /*allow_multi_row=*/true);
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
        // Internal P override: also reflected in the derived split_config fields
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
        split_config.local_grid_y,
        // Rectangle count is program structure (kernel core placement); row
        // distribution within a fixed rectangle layout stays runtime-only.
        split_config.num_rects,
        // Compute-body mode, single-sourced with the factory's kernel-define
        // selection (see compute_body_mode). For k >= 1024 the mode is
        // width-independent (one segmented codepath at every width), so the
        // hash carries NO width term there -- growing-prefill callers never
        // recompile crossing the old 65536 fused boundary. For smaller k the
        // mode still folds in the <= 32-chunk fused bit, the only width term.
        static_cast<uint32_t>(program::compute_body_mode(attrs.k, input.logical_shape()[-1])));
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
    if (attrs.row_count.has_value()) {
        // Row-window launch: the output carries only the window's rows.
        output_shape_vec[output_shape_vec.size() - 2] = *attrs.row_count;
    }
    const ttnn::Shape output_shape(std::move(output_shape_vec));

    const auto memory_config = tensor_args.input_tensor.memory_config();
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config));
}

tensor_return_value_t TopkLargeIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<TopkLargeIndicesDeviceOperation::operation_attributes_t, TopkLargeIndicesDeviceOperation::tensor_args_t>
TopkLargeIndicesDeviceOperation::invoke(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    std::optional<uint32_t> num_slices,
    std::optional<uint32_t> row_start,
    std::optional<uint32_t> row_count) {
    return {
        operation_attributes_t{
            .k = k,
            .valid_length = valid_length,
            .num_slices = num_slices,
            .row_start = row_start,
            .row_count = row_count},
        tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

namespace {

// Hybrid row split: for canonical multi-row calls whose rows exceed the worker
// grid (>= 2 row-parallel waves), peel the partially-filled last wave off into
// a concurrent multi-rectangle launch — the row-parallel full waves keep every
// core busy, and the remainder rows run column-parallel trees on the cores the
// last wave would have left idle. Two launches over one un-sliced input (the
// device op's internal row window), then a cheap [rows, k] concat. Returns
// (full-wave rows, remainder rows, remainder P), or nullopt when the plain
// single launch is already the right program.
struct HybridSplit {
    uint32_t full_wave_rows;
    uint32_t remainder_rows;
    uint32_t remainder_slices;
};
std::optional<HybridSplit> hybrid_row_split(
    const Tensor& input, uint32_t k, std::optional<uint32_t> num_slices) {
    if (num_slices.has_value()) {
        // An internal caller already chose an explicit P: keep its single launch.
        return std::nullopt;
    }
    if (input.storage_type() != StorageType::DEVICE || input.device() == nullptr) {
        return std::nullopt;
    }
    const auto& shape = input.logical_shape();
    if (shape.rank() < 2) {
        return std::nullopt;
    }
    const uint32_t rows = operations::experimental::topk_large_indices::flattened_rows_excluding_last_dim(shape);
    if (rows == 0 || rows != shape[shape.rank() - 2]) {
        return std::nullopt;  // the row window needs the canonical [1.., R, W] shape
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t cores = static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
    if (cores == 0 || rows <= cores) {
        return std::nullopt;  // single row-parallel wave (or the model's own rect pick) already optimal
    }
    const uint32_t waves = tt::div_up(rows, cores);
    const uint32_t r1 = cores * (waves - 1);
    const uint32_t r2 = rows - r1;
    // Split only when the remainder genuinely takes (and wins on) the
    // multi-rectangle path; otherwise the extra launch + concat is pure cost.
    // Model the hybrid structure on the PHYSICAL width, matching the regular
    // column-parallel path. valid_length is a runtime-only search bound: it
    // rebalances active chunks inside the already-selected slices and must not
    // select a different factory, P, or program hash as a prefix grows.
    const uint32_t n = shape[shape.rank() - 1];
    const auto cfg = operations::experimental::topk_large_indices::program::compute_column_split_config(
        k, n, r2, grid, std::nullopt, /*allow_multi_row=*/true);
    if (!cfg.enabled || cfg.num_rects < 2) {
        return std::nullopt;
    }
    // Pin the remainder P explicitly so its row-window launch uses exactly the
    // physical-width split selected here.
    return HybridSplit{r1, r2, cfg.num_slices};
}

}  // namespace

Tensor topk_large_indices(const Tensor& input_tensor, uint32_t k, std::optional<uint32_t> valid_length) {
    using Op = operations::experimental::topk_large_indices::TopkLargeIndicesDeviceOperation;
    if (const auto split = hybrid_row_split(input_tensor, k, std::nullopt)) {
        auto run = [&](uint32_t start, uint32_t count, std::optional<uint32_t> window_slices) {
            auto [attrs, args] = Op::invoke(input_tensor, k, valid_length, window_slices, start, count);
            return ttnn::device_operation::launch<Op>(attrs, args);
        };
        Tensor full_waves = run(0, split->full_wave_rows, std::nullopt);
        Tensor remainder = run(split->full_wave_rows, split->remainder_rows, split->remainder_slices);
        const int rows_dim = static_cast<int>(input_tensor.logical_shape().rank()) - 2;
        return ttnn::concat(
            std::vector<Tensor>{full_waves, remainder}, rows_dim, input_tensor.memory_config());
    }
    auto [operation_attributes, tensor_args] = Op::invoke(input_tensor, k, valid_length);
    return ttnn::device_operation::launch<Op>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
