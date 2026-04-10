// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/overloaded.hpp>

#include "ttnn/tensor/types.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/memory_config/memory_config.hpp"

#include <tracy/Tracy.hpp>

namespace tt::tt_metal {

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

bool is_cpu_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::HOST; }

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

CBDescriptor cb_descriptor_from_sharded_tensor(
    uint8_t cb_index,
    const Tensor& tensor,
    uint32_t address_offset,
    uint32_t total_size,
    const std::optional<CoreRangeSet>& core_ranges) {
    TT_FATAL(tensor.is_sharded(), "Tensor must be sharded to automatically create a CBDescriptor");
    TT_FATAL(
        (address_offset + total_size) <= tensor.buffer()->aligned_size_per_bank(),
        "Address offset + total size exceeds buffer size");

    uint32_t effective_total_size = (total_size != 0) ? total_size : tensor.buffer()->aligned_size_per_bank();

    return CBDescriptor{
        .total_size = effective_total_size,
        .core_ranges = core_ranges.value_or(tensor.shard_spec()->grid),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = datatype_to_dataformat_converter(tensor.tensor_spec().tensor_layout().get_data_type()),
            .page_size = tensor.buffer()->aligned_page_size(),
            .tile = TileDescriptor(tensor.tensor_spec().tile())}},
        .buffer = tensor.buffer(),
        .address_offset = address_offset,
        .global_circular_buffer = nullptr};
}

std::vector<CoreCoord> get_optimal_worker_cores_for_sharded_tensor(const Tensor& tensor, NOC noc) {
    /**
    This function takes in a sharded device tensor (can be legacy 2D sharded or ND sharded) and returns the optimal
    worker cores to launch programs on for the tensor.

    If the tensor is L1 sharded, the function returns a vector of CoreCoords of all the cores that have shards on them
    in order (based on if the shard orientation is in row or column major order).

    If the tensor is DRAM sharded, the function returns a vector of CoreCoords in order (based on shard orientation) of
    the optimal worker core for each DRAM bank with shards.

    The intended use for this API is inside sharded program factories to get the optimal worker cores to launch the
    program and kernels on. Since the core grid provided in the shard_spec and nd_shard_spec may be larger than the
    number of shards that exist, not all cores in the core grid will have shards on them. This API returns the cores
    that have shards on them in order (based on shard orientation) so that the program and kernels will not be launched
    on cores with no data on them (this can cause failures).
    **/
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Tensor must be on device to compute optimal worker cores.");
    TT_FATAL(tensor.is_sharded(), "Tensor must be sharded to compute optimal worker cores.");
    if (!tensor.memory_config().is_dram()) {
        return tensor.buffer()->buffer_distribution_spec().value().cores_with_data();
    }
    TT_FATAL(tensor.device() != nullptr, "Device pointer must be valid when selecting optimal DRAM worker cores");
    auto all_dram_workers = tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(noc);
    const auto dram_banks = tensor.buffer()->buffer_distribution_spec().value().cores_with_data();
    std::vector<CoreCoord> ordered_worker_cores_with_data;
    ordered_worker_cores_with_data.reserve(dram_banks.size());
    for (const auto& dram_core : dram_banks) {
        const uint32_t dram_channel = tensor.device()->dram_channel_from_logical_core(dram_core);
        ordered_worker_cores_with_data.push_back(all_dram_workers[dram_channel]);
    }
    return ordered_worker_cores_with_data;
}

MemoryConfig compute_auto_shard_spec(const Tensor& input_tensor, const MemoryConfig& output_memory_config) {
    // If output memory config is not sharded or already has a shard_spec, return as-is
    if (!output_memory_config.is_sharded() || output_memory_config.shard_spec().has_value()) {
        return output_memory_config;
    }

    // If input tensor has a shard_spec, reuse it for the output
    if (input_tensor.is_sharded() && input_tensor.shard_spec().has_value()) {
        return output_memory_config.with_shard_spec(input_tensor.shard_spec().value());
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE && input_tensor.device() != nullptr,
        "Input tensor must be on device for automatic shard spec computation");

    const auto& input_shape = input_tensor.padded_shape();
    const auto* device = input_tensor.device();
    const auto memory_layout = output_memory_config.memory_layout();

    // Get device compute grid
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    uint32_t grid_x = static_cast<uint32_t>(grid_size.x);
    uint32_t grid_y = static_cast<uint32_t>(grid_size.y);
    uint32_t max_grid_cores = grid_x * grid_y;

    // Tile alignment constants
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;

    // L1 alignment: compute minimum shard width (in elements) that satisfies L1 byte alignment
    uint32_t l1_alignment = hal::get_l1_alignment();
    DataType input_dtype = input_tensor.dtype();
    tt::DataFormat input_df = tt::tt_metal::datatype_to_dataformat_converter(input_dtype);
    uint32_t datum_size_bytes = datum_size(input_df);
    TT_FATAL(
        datum_size_bytes > 0, "Invalid datum_size_bytes: must be > 0. Data format: {}", static_cast<int>(input_df));
    uint32_t min_width_for_l1 = tt::div_up(l1_alignment, datum_size_bytes);
    uint32_t l1_aligned_tile_w = tt::round_up(min_width_for_l1, TILE_W);

    // Compute 2D tensor dimensions: height = product of all dims except last, width = last dim
    uint32_t total_height = input_tensor.physical_volume() / input_shape[-1];
    uint32_t total_width = input_shape[-1];

    std::array<uint32_t, 2> shard_shape = {0, 0};
    uint32_t num_cores = 0;

    switch (memory_layout) {
        case TensorMemoryLayout::WIDTH_SHARDED: {
            TT_FATAL(
                total_width >= l1_aligned_tile_w,
                "WIDTH_SHARDED: total_width ({}) must be >= l1_aligned_tile_width ({})",
                total_width,
                l1_aligned_tile_w);

            // Find maximum cores that produce even shards (no remainder).
            // Even shards have significantly better performance than uneven ones.
            uint32_t max_possible_cores = std::min(max_grid_cores, total_width / l1_aligned_tile_w);
            max_possible_cores = std::max(1u, max_possible_cores);

            uint32_t best_num_cores = 1;
            uint32_t best_shard_width = tt::round_up(total_width, l1_aligned_tile_w);

            for (uint32_t try_cores = max_possible_cores; try_cores >= 1; --try_cores) {
                uint32_t shard_w = tt::round_up(tt::div_up(total_width, try_cores), l1_aligned_tile_w);
                uint32_t actual_cores = tt::div_up(total_width, shard_w);

                if (actual_cores > max_grid_cores) {
                    continue;
                }

                // Prefer even shards: total_width % shard_w == 0
                bool is_even = (total_width % shard_w == 0);
                bool is_better = (actual_cores > best_num_cores) || (actual_cores == best_num_cores && is_even);

                if (is_better) {
                    best_num_cores = actual_cores;
                    best_shard_width = shard_w;
                    if (is_even && best_num_cores >= max_possible_cores) {
                        break;
                    }
                }
            }

            shard_shape = {total_height, best_shard_width};
            num_cores = best_num_cores;
            break;
        }
        case TensorMemoryLayout::HEIGHT_SHARDED: {
            TT_FATAL(
                total_height >= TILE_H,
                "HEIGHT_SHARDED: total_height ({}) must be >= TILE_HEIGHT ({})",
                total_height,
                TILE_H);

            uint32_t max_possible_cores = std::min(max_grid_cores, total_height / TILE_H);
            max_possible_cores = std::max(1u, max_possible_cores);

            uint32_t best_num_cores = 1;
            uint32_t best_shard_height = tt::round_up(total_height, TILE_H);

            for (uint32_t try_cores = max_possible_cores; try_cores >= 1; --try_cores) {
                uint32_t shard_h = tt::round_up(tt::div_up(total_height, try_cores), TILE_H);
                uint32_t actual_cores = tt::div_up(total_height, shard_h);

                if (actual_cores > max_grid_cores) {
                    continue;
                }

                bool is_even = (total_height % shard_h == 0);
                bool is_better = (actual_cores > best_num_cores) || (actual_cores == best_num_cores && is_even);

                if (is_better) {
                    best_num_cores = actual_cores;
                    best_shard_height = shard_h;
                    if (is_even && best_num_cores >= max_possible_cores) {
                        break;
                    }
                }
            }

            shard_shape = {best_shard_height, total_width};
            num_cores = best_num_cores;
            break;
        }
        case TensorMemoryLayout::BLOCK_SHARDED: {
            // For block sharding with COL_MAJOR orientation:
            // num_shards_height <= grid.x, num_shards_width <= grid.y
            uint32_t best_total_cores = 0;
            uint32_t best_shard_h = 0;
            uint32_t best_shard_w = 0;
            uint32_t best_shards_h = 0;
            uint32_t best_shards_w = 0;
            bool best_is_even = false;

            uint32_t max_shards_h = std::min(grid_x, tt::div_up(total_height, TILE_H));
            uint32_t max_shards_w = std::min(grid_y, tt::div_up(total_width, l1_aligned_tile_w));

            for (uint32_t try_shards_h = max_shards_h; try_shards_h >= 1; --try_shards_h) {
                uint32_t shard_h = tt::round_up(tt::div_up(total_height, try_shards_h), TILE_H);
                shard_h = std::max(shard_h, TILE_H);
                uint32_t actual_shards_h = tt::div_up(total_height, shard_h);
                if (actual_shards_h > grid_x) {
                    continue;
                }

                for (uint32_t try_shards_w = max_shards_w; try_shards_w >= 1; --try_shards_w) {
                    uint32_t shard_w = tt::round_up(tt::div_up(total_width, try_shards_w), l1_aligned_tile_w);
                    shard_w = std::max(shard_w, l1_aligned_tile_w);
                    uint32_t actual_shards_w = tt::div_up(total_width, shard_w);
                    if (actual_shards_w > grid_y) {
                        continue;
                    }

                    uint32_t total_cores = actual_shards_h * actual_shards_w;
                    if (total_cores > max_grid_cores) {
                        continue;
                    }

                    bool is_even = (total_height % shard_h == 0) && (total_width % shard_w == 0);
                    bool is_better = (total_cores > best_total_cores) ||
                                     (total_cores == best_total_cores && is_even && !best_is_even);

                    if (is_better) {
                        best_total_cores = total_cores;
                        best_shard_h = shard_h;
                        best_shard_w = shard_w;
                        best_shards_h = actual_shards_h;
                        best_shards_w = actual_shards_w;
                        best_is_even = is_even;
                    }
                }
            }

            TT_FATAL(
                best_total_cores > 0,
                "BLOCK_SHARDED: Could not find valid shard configuration for tensor {}x{} on grid {}x{}",
                total_height,
                total_width,
                grid_x,
                grid_y);

            shard_shape = {best_shard_h, best_shard_w};

            // For BLOCK_SHARDED, create rectangular grid directly
            CoreRange block_range({0, 0}, {best_shards_h - 1, best_shards_w - 1});
            CoreRangeSet grid_set({block_range});
            ShardSpec shard_spec(grid_set, shard_shape, ShardOrientation::COL_MAJOR);
            return output_memory_config.with_shard_spec(shard_spec);
        }
        default:
            TT_FATAL(
                false,
                "Unsupported memory layout for automatic shard_spec creation: {}",
                static_cast<int>(memory_layout));
    }

    // For WIDTH_SHARDED and HEIGHT_SHARDED: create core range set
    bool row_wise = (memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
    CoreRangeSet grid_set = num_cores_to_corerangeset(num_cores, grid_size, row_wise);

    ShardOrientation orientation = row_wise ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR;
    ShardSpec shard_spec(grid_set, shard_shape, orientation);
    return output_memory_config.with_shard_spec(shard_spec);
}

}  // namespace tt::tt_metal
