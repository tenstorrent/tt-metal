// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/split/device/split_device_operation.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>

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

// 2-level batching cutoff (B=sqrt(N) minimises unique slice_starts); 64 caps cold-JIT ~13 s at N=4096.
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

    // 2-level batching for large equal-chunk splits: total unique compilations = O(N/B + B).
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
    // Default output MC: mirror input; rescale shard_spec per-chunk; DRAM on rescale fail or L1 overflow.
    const MemoryConfig desired_mem_cfg = [&]() -> MemoryConfig {
        const auto& in_mc = input_tensor.memory_config();
        const size_t num_chunks_local = split_sizes.size();
        const int64_t norm_dim_local = input_tensor.logical_shape().get_normalized_index(dim);
        const bool equal_split =
            std::all_of(split_sizes.begin(), split_sizes.end(), [&](auto s) { return s == split_sizes[0]; });

        // Compute per-chunk padded shape used to rescale/synthesize output shard_spec.
        auto per_chunk_padded = [&]() -> std::optional<ttnn::Shape> {
            if (!equal_split) {
                return std::nullopt;
            }
            const auto& in_padded = input_tensor.padded_shape();
            const uint32_t padded_dim = in_padded[norm_dim_local];
            if (padded_dim % static_cast<uint32_t>(num_chunks_local) != 0) {
                return std::nullopt;
            }
            ttnn::SmallVector<uint32_t> v(in_padded.cbegin(), in_padded.cend());
            v[norm_dim_local] = padded_dim / static_cast<uint32_t>(num_chunks_local);
            return ttnn::Shape(v);
        };

        // Fill a missing shard_spec on any sharded MC (mirrors slice::resolve_mc).
        auto synthesize_spec = [&](const MemoryConfig& mc) -> MemoryConfig {
            if (!mc.is_sharded() || mc.shard_spec().has_value()) {
                return mc;
            }
            auto chunk_padded = per_chunk_padded();
            if (chunk_padded.has_value() && in_mc.is_sharded() && in_mc.shard_spec().has_value() &&
                in_mc.memory_layout() == mc.memory_layout()) {
                auto adj = ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape(
                    *in_mc.shard_spec(), input_tensor.padded_shape(), *chunk_padded);
                if (adj.has_value()) {
                    return MemoryConfig(mc.memory_layout(), mc.buffer_type(), adj);
                }
            }
            if (chunk_padded.has_value()) {
                auto synth = ttnn::operations::data_movement::transpose::generate_transpose_shard_spec(
                    input_tensor, *chunk_padded, mc.memory_layout());
                return MemoryConfig(mc.memory_layout(), mc.buffer_type(), synth);
            }
            log_warning(tt::LogOp, "ttnn.split: cannot synthesize shard_spec for sharded MC; downgrading to DRAM.");
            return ttnn::DRAM_MEMORY_CONFIG;
        };

        if (memory_config_arg.has_value()) {
            return synthesize_spec(*memory_config_arg);
        }

        // L1 CB-clash guard: true when N L1 chunks + slice CBs fit L1 on core [0,0].
        auto l1_budget_ok = [&]() -> bool {
            // Skip: DRAM has no CB clash; N ≤ 2 uses the native TILE fast path (single bounded-L1 kernel).
            if (in_mc.buffer_type() != tt::tt_metal::BufferType::L1 || num_chunks_local <= 2) {
                return true;
            }
            auto* dev = input_tensor.device();
            const auto lowest = dev->lowest_occupied_compute_l1_address();
            uint32_t max_l1 = lowest.value_or(dev->l1_size_per_core());
            const uint32_t base = dev->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
            max_l1 = (max_l1 > base) ? (max_l1 - base) : 0;

            // Padded volume + tile_size(DataFormat) so bfp8_b/bfp4_b exponents + TILE padding aren't under-counted.
            uint64_t chunk_bytes = 0;
            uint64_t page_bytes = 0;
            if (input_tensor.layout() == tt::tt_metal::Layout::TILE) {
                const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
                const uint64_t tile_bytes = tt::tile_size(data_format);
                const uint64_t total_tiles =
                    input_tensor.physical_volume() / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
                chunk_bytes = (total_tiles / static_cast<uint64_t>(num_chunks_local)) * tile_bytes;
                page_bytes = tile_bytes;
            } else {
                const uint64_t elem = input_tensor.element_size();
                chunk_bytes = input_tensor.physical_volume() / static_cast<uint64_t>(num_chunks_local) * elem;
                page_bytes = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH * elem;
            }
            // CB overhead scales with N: each slice call stamps its own reader+writer CBs on core [0,0].
            constexpr uint64_t CB_PAGES_PER_SLICE = 4;
            const uint64_t cb_overhead_bytes =
                static_cast<uint64_t>(num_chunks_local) * CB_PAGES_PER_SLICE * page_bytes;
            const uint64_t est_bytes = chunk_bytes * num_chunks_local + cb_overhead_bytes;

            if (est_bytes > static_cast<uint64_t>(max_l1)) {
                log_warning(
                    tt::LogOp,
                    "ttnn.split: L1 budget exceeded (need ~{} B, have {} B for {} chunks); DRAM downgrade.",
                    est_bytes,
                    max_l1,
                    num_chunks_local);
                return false;
            }
            return true;
        };

        // Sharded input, no user MC: adjust input's spec if present, else DRAM (nd-sharded has no legacy spec).
        if (in_mc.is_sharded()) {
            auto chunk_padded = per_chunk_padded();
            if (!chunk_padded.has_value()) {
                log_warning(tt::LogOp, "ttnn.split: unequal/non-divisible split on sharded input; DRAM downgrade.");
                return ttnn::DRAM_MEMORY_CONFIG;
            }
            if (!in_mc.shard_spec().has_value()) {
                // ND-sharded: no legacy spec, and generate_transpose_shard_spec can't emit ND geometry.
                log_warning(tt::LogOp, "ttnn.split: nd-sharded input (no legacy shard_spec); DRAM downgrade.");
                return ttnn::DRAM_MEMORY_CONFIG;
            }
            auto derived = ttnn::operations::data_movement::transpose::adjust_shard_spec_to_shape(
                *in_mc.shard_spec(), input_tensor.padded_shape(), *chunk_padded);
            if (!derived.has_value()) {
                log_warning(tt::LogOp, "ttnn.split: shard_spec can't be rescaled to per-chunk shape; DRAM downgrade.");
                return ttnn::DRAM_MEMORY_CONFIG;
            }
            if (input_tensor.layout() == tt::tt_metal::Layout::TILE &&
                (derived->shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                 derived->shape[1] % tt::constants::TILE_WIDTH != 0)) {
                log_warning(
                    tt::LogOp,
                    "ttnn.split: derived shard [{}, {}] not tile-aligned; DRAM downgrade.",
                    derived->shape[0],
                    derived->shape[1]);
                return ttnn::DRAM_MEMORY_CONFIG;
            }
            if (!l1_budget_ok()) {
                return ttnn::DRAM_MEMORY_CONFIG;
            }
            return MemoryConfig(in_mc.memory_layout(), in_mc.buffer_type(), derived);
        }

        if (!l1_budget_ok()) {
            return ttnn::DRAM_MEMORY_CONFIG;
        }
        return in_mc;
    }();

    TT_FATAL(!split_sizes.empty(), "split_sizes must not be empty");
    TT_FATAL(
        std::all_of(split_sizes.begin(), split_sizes.end(), [](const auto& x) { return x > 0; }),
        "split_size should be greater than 0, instead got: {}",
        split_sizes);

    // Sharded I/O handled natively by device kernels and slice via TensorAccessor (no de-shard).
    const Tensor& working_input = input_tensor;
    const MemoryConfig& op_mem_cfg = desired_mem_cfg;

    const auto& input_shape = working_input.logical_shape();
    int64_t normalized_dim = input_shape.get_normalized_index(dim);
    std::vector<ttnn::Tensor> results;

    const uint32_t num_chunks = static_cast<uint32_t>(split_sizes.size());

    // Equal N-way split fully covering the dim.
    bool is_equal_n_way_split =
        std::all_of(split_sizes.begin(), split_sizes.end(), [&](auto s) { return s == split_sizes[0]; }) &&
        split_sizes[0] * static_cast<int64_t>(num_chunks) == static_cast<int64_t>(input_shape[normalized_dim]);

    // TILE fast path (native N-chunk kernel): equal split on last dim, TILE, rank≥2, ≥2 tiles/dim, N-divisible, fits
    // grid, shape4d[0]==1 for N>2; else N slice calls.
    tt::tt_metal::IDevice* device = working_input.device();
    const uint32_t grid_dim_x = device->compute_with_storage_grid_size().x;
    const uint32_t grid_dim_y = device->compute_with_storage_grid_size().y;

    // Compute the 4D shape the kernel sees: rank<4 pads with 1s, rank>4 merges leading dims into [0].
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

    // num_cores_z = shape4d[1] (N==2 collapses batch via reshape → z = B*C); must fit grid_dim_x.
    const uint32_t z_4d = (num_chunks == detail::TWO_CHUNKS) ? shape4d[0] * shape4d[1] : shape4d[1];
    bool fits_in_core_grid = input_shape.rank() >= 2 && z_4d < grid_dim_x + 1;

    // Program factory requires padded tile count in split dim divisible by num_chunks.
    uint32_t padded_tiles_in_split_dim = working_input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    bool tiles_divisible_by_chunks = (padded_tiles_in_split_dim % num_chunks == 0);

    // prim::split hard-requires shape4d[0]==1 for N>2 (N==2 handles batch>1 via reshape).
    bool batch_ok_for_n_gt2 = (num_chunks == detail::TWO_CHUNKS) || (shape4d[0] == 1);

    // Guard against grid_dim_y/num_chunks == 0 (invalid CoreRange in program factory).
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
        // N>2: prim::split directly. N==2: split_last_dim_two_chunks_tiled handles batch>1 reshape.
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
        // Slice fallback: if outputs downgraded to DRAM, migrate L1 input too so per-slice CBs don't clash.
        Tensor slice_input = working_input;
        if (slice_input.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 &&
            op_mem_cfg.buffer_type() == tt::tt_metal::BufferType::DRAM && split_sizes.size() > 2) {
            const MemoryConfig dram_interleaved{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
            // Padded volume + tile_size(DataFormat) so bfp8_b/bfp4_b exponents are counted.
            uint64_t input_bytes = 0;
            if (slice_input.layout() == tt::tt_metal::Layout::TILE) {
                const auto df = tt::tt_metal::datatype_to_dataformat_converter(slice_input.dtype());
                const uint64_t total_tiles =
                    slice_input.physical_volume() / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
                input_bytes = total_tiles * tt::tile_size(df);
            } else {
                input_bytes = slice_input.physical_volume() * slice_input.element_size();
            }
            log_warning(tt::LogOp, "ttnn.split: migrating L1 input ({} B) to DRAM before slice fallback.", input_bytes);
            slice_input = ttnn::to_memory_config(slice_input, dram_interleaved, std::nullopt);
        }
        results = detail::split_with_slice_impl(slice_input, split_sizes, normalized_dim, op_mem_cfg);
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
