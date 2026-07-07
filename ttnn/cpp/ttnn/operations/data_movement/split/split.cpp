// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/split/device/split_device_operation.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"

namespace ttnn {

namespace detail {

constexpr auto TWO_CHUNKS = 2;
constexpr auto RANK_FOUR = 4;

std::vector<Tensor> impl_split_last_dim_two_chunks_tiled(const Tensor& input_tensor, const MemoryConfig& mem_config) {
    return ttnn::prim::split(input_tensor, 2, 3, mem_config);
}

std::vector<Tensor> split_last_dim_two_chunks_tiled(const Tensor& input_tensor, const MemoryConfig& mem_config) {
    const auto& shape = input_tensor.padded_shape();
    const bool pre_post_reshape = shape[0] > 1;

    if (!pre_post_reshape) {
        return impl_split_last_dim_two_chunks_tiled(input_tensor, mem_config);
    }

    const int Y = shape[2], X = shape[3];
    const Tensor& reshaped_tensor =
        ttnn::reshape_on_device(input_tensor, ttsl::SmallVector<int32_t>{1, -1, Y, X}, mem_config);

    auto part_reshaped = impl_split_last_dim_two_chunks_tiled(reshaped_tensor, mem_config);

    std::vector<Tensor> results;
    results.reserve(part_reshaped.size());
    for (auto& part : part_reshaped) {
        results.emplace_back(
            ttnn::reshape_on_device(part, ttsl::SmallVector<int32_t>{-1, (int32_t)shape[1], Y, X / 2}, mem_config));
    }

    return results;
}

// Maximum number of slice calls per level before 2-level batching kicks in.
// Each unique slice_start produces a distinct program cache entry; with N chunks
// that means N cold JIT compilations (~100 ms each).  Batching outer slices into
// groups of SPLIT_BATCH_SIZE reduces unique hashes from O(N) to O(N/B + B),
// minimised at B = sqrt(N).  64 keeps cold-JIT time under ~13 s for N = 4096.
constexpr uint32_t SPLIT_BATCH_SIZE = 64;

std::vector<ttnn::Tensor> split_with_slice_impl(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& split_sizes,
    const int32_t dim,
    const MemoryConfig& memory_config) {
    const auto& input_shape = input_tensor.logical_shape();

    // torch requires split size to sum to dim size but since we are using slice we can be more permissive.
    TT_FATAL(
        std::accumulate(split_sizes.begin(), split_sizes.end(), 0L) >= input_shape[dim],
        "Split sizes should sum to at least dimension size. Split sizes: {} dimension {}",
        split_sizes,
        input_shape[dim]);

    const auto num_chunks = static_cast<uint32_t>(split_sizes.size());

    // 2-level batching for large equal-chunk splits.
    // All outer slices have unique slice_starts on the original tensor (O(N/B) hashes).
    // All inner slices on same-shape batch tensors share the same B program hashes
    // across every outer batch, so total unique compilations = O(N/B + B).
    const bool all_equal = num_chunks > 1 && std::all_of(split_sizes.begin(), split_sizes.end(), [&](auto s) {
                               return s == split_sizes[0];
                           });

    if (all_equal && num_chunks > SPLIT_BATCH_SIZE) {
        const int64_t chunk_size = split_sizes[0];
        const uint32_t n_batches = (num_chunks + SPLIT_BATCH_SIZE - 1) / SPLIT_BATCH_SIZE;

        std::vector<ttnn::Tensor> results;
        results.reserve(num_chunks);

        const ttsl::SmallVector<int64_t> steps(input_shape.rank(), 1);
        ttsl::SmallVector<int64_t> begins(input_shape.rank(), 0);
        ttsl::SmallVector<int64_t> ends(input_shape.cbegin(), input_shape.cend());
        const ttsl::Span<const int64_t> ssteps(steps);

        for (uint32_t b = 0; b < n_batches; b++) {
            const uint32_t actual_batch = std::min(SPLIT_BATCH_SIZE, num_chunks - b * SPLIT_BATCH_SIZE);
            begins[dim] = static_cast<int64_t>(b) * SPLIT_BATCH_SIZE * chunk_size;
            ends[dim] = std::min(
                begins[dim] + static_cast<int64_t>(actual_batch) * chunk_size, static_cast<int64_t>(input_shape[dim]));

            const ttsl::Span<const int64_t> sbegins(begins), sends(ends);
            auto batch_tensor = ttnn::slice(input_tensor, sbegins, sends, ssteps, memory_config);

            // actual_batch <= SPLIT_BATCH_SIZE so this recurse goes to the flat loop below.
            const ttsl::SmallVector<int64_t> inner_sizes(actual_batch, chunk_size);
            auto inner = split_with_slice_impl(batch_tensor, inner_sizes, dim, memory_config);
            results.insert(results.end(), inner.begin(), inner.end());
        }

        return results;
    }

    std::vector<ttnn::Tensor> results;
    results.reserve(split_sizes.size());

    // int64_t coordinates (matching the batching path above) for large-dim consistency.
    const ttsl::SmallVector<int64_t> steps(input_shape.rank(), 1);
    ttsl::SmallVector<int64_t> begins(input_shape.rank(), 0), ends(input_shape.cbegin(), input_shape.cend());
    const ttsl::Span<const int64_t> sbegins(begins), ssteps(steps), sends(ends);

    ends[dim] = 0;
    for (const auto& s : split_sizes) {
        ends[dim] = std::min(ends[dim] + s, static_cast<int64_t>(input_shape[dim]));
        results.emplace_back(ttnn::slice(input_tensor, sbegins, sends, ssteps, memory_config));
        begins[dim] += s;
    }

    return results;
}
}  // namespace detail

std::vector<ttnn::Tensor> split(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& split_sizes,
    int64_t dim,
    const std::optional<MemoryConfig>& memory_config_arg) {
    // Capture the desired output memory config.
    // When no explicit config is given, mirror the input's memory config — except for a
    // sharded input, where the per-chunk output shape generally won't fit the input's shard
    // spec, so default to DRAM interleaved. An explicit sharded output config is honored and
    // produced natively by the device kernel (its shard spec must match the chunk shape).
    const MemoryConfig desired_mem_cfg = [&]() -> MemoryConfig {
        if (memory_config_arg.has_value()) {
            return *memory_config_arg;
        }
        if (input_tensor.memory_config().is_sharded()) {
            return ttnn::DRAM_MEMORY_CONFIG;
        }
        return input_tensor.memory_config();
    }();

    TT_FATAL(!split_sizes.empty(), "split_sizes must not be empty");
    TT_FATAL(
        std::all_of(split_sizes.begin(), split_sizes.end(), [](const auto& x) { return x > 0; }),
        "split_size should be greater than 0, instead got: {}",
        split_sizes);

    // Native sharded I/O: the device kernels (and ttnn::slice in the fallback) read and write
    // sharded buffers directly via TensorAccessor, so no host-side conversion is needed.
    const Tensor& working_input = input_tensor;
    const MemoryConfig& op_mem_cfg = desired_mem_cfg;

    const auto& input_shape = working_input.logical_shape();

    // Normalize negative dimension to positive index
    int64_t normalized_dim = input_shape.get_normalized_index(dim);

    std::vector<ttnn::Tensor> results;

    const uint32_t num_chunks = static_cast<uint32_t>(split_sizes.size());

    // True when all requested chunk sizes are equal and together cover the full dimension.
    bool is_equal_n_way_split =
        std::all_of(split_sizes.begin(), split_sizes.end(), [&](auto s) { return s == split_sizes[0]; }) &&
        split_sizes[0] * static_cast<int64_t>(num_chunks) == static_cast<int64_t>(input_shape[normalized_dim]);

    // ---------------------------------------------------------------------------
    // Fast path 1: TILE layout, equal N-way split on the last dim.
    // Uses the native N-chunk TILE device kernel (single-pass, no slice overhead).
    //
    // The native TILE kernel is selected only when ALL of the following hold (see
    // can_use_tile_kernel below); otherwise the split falls back to N independent
    // ttnn::slice calls (split_with_slice_impl), which handle every other case
    // (ROW_MAJOR, unequal sizes, non-last-dim, batch>1 for N>2, etc.):
    //   - equal N-way split that exactly covers the dimension (is_equal_n_way_split)
    //   - split dim is the last dim (normalized_dim == rank - 1)
    //   - input is TILE layout, rank >= 2
    //   - at least 2 tiles in each of the last two dims
    //   - padded tile count in the split dim is divisible by num_chunks
    //   - it all fits the core grid: z_4d <= grid_dim_x and num_chunks <= grid_dim_y
    //   - shape4d[0] == 1 for N>2 (the N==2 path collapses batch via reshape instead)
    // Sharded input/output are handled natively on both paths (no de-shard step).
    // ---------------------------------------------------------------------------
    tt::tt_metal::IDevice* device = working_input.device();
    const uint32_t grid_dim_x = device->compute_with_storage_grid_size().x;
    const uint32_t grid_dim_y = device->compute_with_storage_grid_size().y;

    // Compute the 4D shape the device kernel will see (call squeeze_shape_to_4D once).
    // For rank < 4: unsqueeze_to_4D prepends 1s → shape4d = [1...1, dims...]
    // For rank == 4: identity
    // For rank > 4: squeeze_shape_to_4D merges the leading (rank-3) dims into shape4d[0]
    std::array<uint32_t, 4> shape4d = {1, 1, 1, 1};
    if (input_shape.rank() > static_cast<size_t>(detail::RANK_FOUR)) {
        const auto s4 = ttnn::operations::data_movement::squeeze_shape_to_4D(working_input.logical_shape());
        for (int i = 0; i < 4; i++) {
            shape4d[i] = s4[i];
        }
    } else {
        for (size_t i = 0; i < input_shape.rank(); i++) {
            shape4d[4 - input_shape.rank() + i] = input_shape[i];
        }
    }

    // The program factory sets num_cores_z = shape4d[1] (z dimension).
    // For N==2, split_last_dim_two_chunks_tiled reshapes [B,C,H,W]→[1,B*C,H,W], so z = B*C.
    // For N>2, prim::split receives shape4d directly, so z = shape4d[1].
    // num_cores_r = num_cores_x * num_cores_z must fit in grid_dim_x.
    const uint32_t z_4d = (num_chunks == detail::TWO_CHUNKS) ? shape4d[0] * shape4d[1] : shape4d[1];
    bool fits_in_core_grid = input_shape.rank() >= 2 && z_4d < grid_dim_x + 1;

    // The program factory requires the padded tile count in the split dim to be
    // divisible by num_chunks (otherwise the Y-core grouping doesn't work out).
    uint32_t padded_tiles_in_split_dim = working_input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    bool tiles_divisible_by_chunks = (padded_tiles_in_split_dim % num_chunks == 0);

    // prim::split hard-requires shape4d[0]==1 (see validate_on_program_cache_miss).
    // N==2 handles batch>1 via reshape; N>2 requires shape4d[0]==1 already.
    bool batch_ok_for_n_gt2 = (num_chunks == detail::TWO_CHUNKS) || (shape4d[0] == 1);

    // The program factory groups Y-cores into num_chunks equal groups.
    // If num_chunks > grid_dim_y, max_cores_per_chunk = grid_dim_y/num_chunks = 0, producing an
    // invalid CoreRange. Guard against this.
    bool chunks_fit_in_y_grid = (num_chunks <= grid_dim_y);

    bool can_use_tile_kernel = is_equal_n_way_split && normalized_dim == static_cast<int64_t>(input_shape.rank()) - 1 &&
                               working_input.layout() == Layout::TILE && input_shape.rank() >= 2 && fits_in_core_grid &&
                               input_shape[-2] / tt::constants::TILE_HEIGHT >= 2 &&
                               input_shape[-1] / tt::constants::TILE_WIDTH >= 2 && tiles_divisible_by_chunks &&
                               batch_ok_for_n_gt2 && chunks_fit_in_y_grid;

    if (can_use_tile_kernel) {
        ttnn::Tensor input_tensor_4d;
        if (input_shape.rank() > detail::RANK_FOUR) {
            input_tensor_4d = operations::data_movement::squeeze_from_ND_to_4D(working_input);
        } else if (input_shape.rank() < detail::RANK_FOUR) {
            input_tensor_4d = unsqueeze_to_4D(working_input);
        } else {
            input_tensor_4d = working_input;
        }
        // For N>2, use prim::split directly (generalised factory).
        // For N==2, split_last_dim_two_chunks_tiled handles the batch>1 reshape.
        std::vector<Tensor> outputs_4d;
        if (num_chunks == detail::TWO_CHUNKS) {
            outputs_4d = detail::split_last_dim_two_chunks_tiled(input_tensor_4d, op_mem_cfg);
        } else {
            outputs_4d = ttnn::prim::split(input_tensor_4d, static_cast<int>(num_chunks), 3, op_mem_cfg);
        }
        results.reserve(num_chunks);
        for (const auto& t : outputs_4d) {
            ttsl::SmallVector<uint32_t> final_shape(input_shape.cbegin(), input_shape.cend());
            final_shape.back() = t.logical_shape()[-1];
            results.emplace_back(ttnn::view(t, ttnn::Shape(final_shape)));
        }
    } else {
        // Fallback: unequal splits, ROW_MAJOR, batch>1 for N>2, or anything not
        // handled by the TILE fast path above. Uses N independent slice calls.
        results = detail::split_with_slice_impl(working_input, split_sizes, normalized_dim, op_mem_cfg);
    }

    return results;
}

std::vector<ttnn::Tensor> split(
    const ttnn::Tensor& input_tensor,
    int64_t split_size,
    int64_t dim,
    const std::optional<MemoryConfig>& memory_config_arg) {
    TT_FATAL(split_size > 0, "split_size must be greater than 0, but got: {}", split_size);

    // Normalize negative dimension to positive index
    const auto& input_shape = input_tensor.logical_shape();
    int64_t normalized_dim = input_shape.get_normalized_index(dim);

    const auto num_chunks = std::ceil(static_cast<float>(input_shape[normalized_dim]) / static_cast<float>(split_size));

    const ttsl::SmallVector<int64_t> split_sizes(num_chunks, split_size);
    // Pass memory_config_arg directly so the split_sizes overload applies the
    // same sharding-default logic (sharded input → DRAM output when no explicit config).
    return ttnn::split(input_tensor, split_sizes, normalized_dim, memory_config_arg);
}

}  // namespace ttnn
