// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_factory_utils.hpp"

#include <algorithm>
#include <array>
#include <enchantum/enchantum.hpp>
#include <set>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

namespace ttnn::experimental::prim::kda_factory_detail {

void check_allocated_device_tensor(
    const Tensor& tensor, std::string_view operation_name, std::string_view tensor_name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "{}: {} must be an allocated device tensor",
        operation_name,
        tensor_name);
}

void check_layout(
    const Tensor& tensor,
    tt::tt_metal::Layout required_layout,
    std::string_view operation_name,
    std::string_view tensor_name) {
    TT_FATAL(
        tensor.layout() == required_layout,
        "{}: {} must use {} layout, got {}",
        operation_name,
        tensor_name,
        enchantum::to_string(required_layout),
        tensor.layout());
}

void check_dtype(
    const Tensor& tensor,
    tt::tt_metal::DataType required_dtype,
    std::string_view operation_name,
    std::string_view tensor_name) {
    TT_FATAL(
        tensor.dtype() == required_dtype,
        "{}: {} must be {}, got {}",
        operation_name,
        tensor_name,
        enchantum::to_string(required_dtype),
        tensor.dtype());
}

void check_dtype_in(
    const Tensor& tensor,
    std::span<const tt::tt_metal::DataType> accepted_dtypes,
    std::string_view accepted_dtype_names,
    std::string_view operation_name,
    std::string_view tensor_name) {
    TT_FATAL(!accepted_dtypes.empty(), "{}: accepted dtype set must not be empty", operation_name);
    TT_FATAL(
        std::find(accepted_dtypes.begin(), accepted_dtypes.end(), tensor.dtype()) != accepted_dtypes.end(),
        "{}: {} must be {}, got {}",
        operation_name,
        tensor_name,
        accepted_dtype_names,
        tensor.dtype());
}

void check_matching_dtype(
    const Tensor& lhs, const Tensor& rhs, std::string_view operation_name, std::string_view tensor_group_name) {
    TT_FATAL(
        lhs.dtype() == rhs.dtype(),
        "{}: {} must have matching dtypes, got {} and {}",
        operation_name,
        tensor_group_name,
        lhs.dtype(),
        rhs.dtype());
}

void check_same_device(
    const Tensor& reference,
    const Tensor& candidate,
    std::string_view operation_name,
    std::string_view candidate_name) {
    TT_FATAL(
        candidate.device() == reference.device(),
        "{}: {} must be on the same device as the other inputs",
        operation_name,
        candidate_name);
}

void check_interleaved(const Tensor& tensor, std::string_view operation_name, std::string_view tensor_name) {
    TT_FATAL(!tensor.is_sharded(), "{}: {} must use interleaved memory", operation_name, tensor_name);
}

void check_actual_start(const Tensor& reference, const Tensor& actual_start, std::string_view operation_name) {
    check_allocated_device_tensor(actual_start, operation_name, "actual_start");
    check_layout(actual_start, tt::tt_metal::Layout::ROW_MAJOR, operation_name, "actual_start");
    check_dtype(actual_start, tt::tt_metal::DataType::UINT32, operation_name, "actual_start");
    check_interleaved(actual_start, operation_name, "actual_start");
    check_same_device(reference, actual_start, operation_name, "actual_start");
    TT_FATAL(actual_start.logical_shape().volume() == 1, "{}: actual_start must be a scalar", operation_name);
}

void check_output_interleaved(const tt::tt_metal::MemoryConfig& memory_config, std::string_view operation_name) {
    TT_FATAL(
        !memory_config.is_sharded(),
        "{}: output memory layout must be INTERLEAVED, got {}",
        operation_name,
        enchantum::to_string(memory_config.memory_layout()));
}

void check_summary_memory_layout(
    const tt::tt_metal::MemoryConfig& memory_config, std::string_view operation_name, std::string_view tensor_name) {
    const auto memory_layout = memory_config.memory_layout();
    TT_FATAL(
        memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
            memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
            memory_layout == tt::tt_metal::TensorMemoryLayout::ND_SHARDED,
        "{}: {} must use interleaved, height-sharded or ND-sharded memory, got {}",
        operation_name,
        tensor_name,
        enchantum::to_string(memory_layout));
}

PackedSummaryShape check_packed_summaries(const Tensor& summaries, std::string_view operation_name) {
    constexpr std::array accepted_dtypes = {tt::tt_metal::DataType::FLOAT32, tt::tt_metal::DataType::BFLOAT16};
    check_allocated_device_tensor(summaries, operation_name, "summaries");
    check_layout(summaries, tt::tt_metal::Layout::TILE, operation_name, "summaries");
    check_dtype_in(summaries, accepted_dtypes, "FLOAT32 or BFLOAT16", operation_name, "summaries");
    check_summary_memory_layout(summaries.memory_config(), operation_name, "summaries");
    const auto& shape = summaries.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: summaries must be rank 4 [S, N, K, K+V]", operation_name);
    TT_FATAL(shape[0] == 1 || shape[0] == 2, "{}: summaries must hold 1 or 2 slots, got {}", operation_name, shape[0]);
    TT_FATAL(shape[1] > 0, "{}: summaries must hold at least one transform", operation_name);
    const uint32_t key_dim = shape[2];
    TT_FATAL(
        key_dim > 0 && shape[3] > key_dim && key_dim % tt::constants::TILE_WIDTH == 0 &&
            shape[3] % tt::constants::TILE_WIDTH == 0,
        "{}: summary rows must be [A | B] with positive, tile-aligned K and V",
        operation_name);
    return {
        .slots = static_cast<uint32_t>(shape[0]),
        .transforms = static_cast<uint32_t>(shape[1]),
        .key_dim = key_dim,
        .value_dim = static_cast<uint32_t>(shape[3]) - key_dim};
}

void check_compute_config(const DeviceComputeKernelConfig& config, std::string_view operation_name) {
    TT_FATAL(
        !config.packer_l1_acc,
        "{}: packer_l1_acc=true is unsupported because the compute kernel does not accumulate through L1",
        operation_name);
}

KdaPrepWorkDist distribute_prep(
    tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t core_cap, std::optional<tt::tt_metal::CoreCoord> excluded) {
    const uint32_t grid_cores = grid.x * grid.y;
    const bool skips = excluded.has_value() && excluded->x < grid.x && excluded->y < grid.y;
    const uint32_t max_cores = std::min<uint32_t>(grid_cores - (skips ? 1u : 0u), core_cap);
    const uint32_t count = std::min(total, max_cores);
    TT_FATAL(count > 0, "KDA work distribution needs at least one item (total={})", total);
    const uint32_t base = total / count;
    const uint32_t remainder = total % count;

    KdaPrepWorkDist distribution;
    distribution.cores.reserve(count);
    distribution.wi_start.reserve(count);
    distribution.wi_count.reserve(count);
    std::vector<tt::tt_metal::CoreRange> ranges;
    uint32_t offset = 0;
    for (uint32_t position = 0; distribution.cores.size() < count; ++position) {
        const tt::tt_metal::CoreCoord core{position % grid.x, position / grid.x};
        if (skips && core == *excluded) {
            continue;
        }
        const uint32_t index = distribution.cores.size();
        const uint32_t item_count = base + (index < remainder ? 1u : 0u);
        distribution.cores.push_back(core);
        distribution.wi_start.push_back(offset);
        distribution.wi_count.push_back(item_count);
        offset += item_count;
        // Extend the row's current run, or start a new one.
        if (!ranges.empty() && ranges.back().end_coord.y == core.y && ranges.back().end_coord.x + 1 == core.x) {
            ranges.back().end_coord = core;
        } else {
            ranges.emplace_back(core, core);
        }
    }
    distribution.core_set = skips ? tt::tt_metal::CoreRangeSet(ranges).merge_ranges()
                                  : tt::tt_metal::num_cores_to_corerangeset(count, grid, true);
    return distribution;
}

ValueBlockDistribution distribute_value_blocks(
    tt::tt_metal::CoreCoord grid, uint32_t batch_heads, uint32_t value_tiles) {
    TT_FATAL(batch_heads <= grid.x * grid.y, "KDA heads {} exceed compute cores {}", batch_heads, grid.x * grid.y);
    // The most blocks each head can split into when its blocks share a line of `line` cores and there are `lines`.
    const auto best_split = [&](uint32_t line, uint32_t lines) {
        for (uint32_t candidate = std::min(value_tiles, line); candidate > 1; --candidate) {
            if (value_tiles % candidate == 0 && batch_heads <= (line / candidate) * lines) {
                return candidate;
            }
        }
        return 1u;
    };
    // Per-column math is identical for any split, but more blocks mean fewer value tiles and less L1 per core, so
    // keep the largest split. On a tie a head's blocks take one grid column: the worker rows are physically
    // contiguous, while a grid row can straddle non-worker columns, and value block 0's multicast over such a gap was
    // intermittently twice as slow.
    const uint32_t row_blocks = best_split(grid.x, grid.y);
    const bool vertical = row_blocks > 1 && best_split(grid.y, grid.x) == row_blocks;
    const uint32_t value_blocks = row_blocks;
    const uint32_t line = vertical ? grid.y : grid.x;
    const uint32_t heads_per_line = line / value_blocks;
    ValueBlockDistribution result;
    result.value_blocks = value_blocks;
    result.value_tiles_per_core = value_tiles / value_blocks;
    const auto at = [&](uint32_t along, uint32_t across) {
        return vertical ? tt::tt_metal::CoreCoord{across, along} : tt::tt_metal::CoreCoord{along, across};
    };
    std::vector<tt::tt_metal::CoreRange> lanes;
    for (uint32_t lane = 0; lane * heads_per_line < batch_heads; ++lane) {
        const uint32_t lane_heads = std::min(heads_per_line, batch_heads - lane * heads_per_line);
        lanes.emplace_back(at(0, lane), at(lane_heads * value_blocks - 1, lane));
    }
    for (uint32_t head = 0; head < batch_heads; ++head) {
        for (uint32_t block = 0; block < value_blocks; ++block) {
            result.cores.push_back(at((head % heads_per_line) * value_blocks + block, head / heads_per_line));
            result.head.push_back(head);
            result.value_block.push_back(block);
        }
    }
    // Merged ranges keep dispatch to one multicast when the lanes fill the same span.
    result.core_set = tt::tt_metal::CoreRangeSet(lanes).merge_ranges();
    return result;
}

uint32_t tile_bytes(const Tensor& tensor) {
    return tensor.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype()));
}

tt::tt_metal::CoreRangeSet merged_core_set(std::span<const tt::tt_metal::CoreCoord> cores) {
    return tt::tt_metal::CoreRangeSet(ttsl::Span<const tt::tt_metal::CoreCoord>(cores.data(), cores.size()))
        .merge_ranges();
}

bool same_cores(const tt::tt_metal::CoreRangeSet& set, std::span<const tt::tt_metal::CoreCoord> cores) {
    const auto set_cores = tt::tt_metal::corerange_to_cores(set);
    return std::set<tt::tt_metal::CoreCoord>(set_cores.begin(), set_cores.end()) ==
           std::set<tt::tt_metal::CoreCoord>(cores.begin(), cores.end());
}

std::vector<tt::tt_metal::CoreCoord> shard_cores(const Tensor& tensor) {
    const auto& shard = *tensor.memory_config().shard_spec();
    return tt::tt_metal::corerange_to_cores(
        shard.grid, std::nullopt, shard.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
}

tt::tt_metal::CBDescriptor cb_descriptor(
    uint32_t index,
    uint32_t pages,
    uint32_t page_size,
    tt::DataFormat format,
    const tt::tt_metal::CoreRangeSet& cores) {
    return tt::tt_metal::CBDescriptor{
        .total_size = pages * page_size,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index, .data_format = format, .page_size = page_size}}},
    };
}

tt::tt_metal::CBDescriptor sharded_cb_descriptor(uint32_t index, const Tensor& tensor) {
    const auto& shard = *tensor.memory_config().shard_spec();
    const uint32_t page = tile_bytes(tensor);
    auto cb = cb_descriptor(
        index,
        shard.shape[0] * shard.shape[1] / tt::constants::TILE_HW,
        page,
        tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype()),
        shard.grid);
    cb.buffer = tensor.buffer();
    return cb;
}

tt::tt_metal::ComputeConfigDescriptor compute_config_descriptor(
    tt::ARCH arch, const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(arch, compute_kernel_config);
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
}

}  // namespace ttnn::experimental::prim::kda_factory_detail
