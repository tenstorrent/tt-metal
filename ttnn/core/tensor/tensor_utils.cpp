// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/overloaded.hpp>

#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/types.hpp"

#include <tracy/Tracy.hpp>

namespace tt {
namespace tt_metal {

tt::tt_metal::Shape infer_dims_for_reshape(const Tensor& tensor, tt::stl::Span<const int32_t> shape) {
    int64_t old_volume = tensor.logical_volume();
    int64_t new_volume = 1;
    int64_t index_of_negative_1 = -1;
    bool has_zero = false;
    for (auto index = 0; index < shape.size(); ++index) {
        if (shape[index] == -1) {
            if (index_of_negative_1 != -1) {
                std::string error_msg = "Shape cannot have more than 1 elements that is set to -1! Shape used: (";
                for (const auto& s : shape) {
                    error_msg += std::to_string(s) + ",";
                }
                error_msg += ")";
                TT_THROW("{}", error_msg);
            }
            index_of_negative_1 = index;
        } else {
            if (shape[index] == 0) {
                has_zero = true;
            }
            new_volume *= shape[index];
        }
    }
    if (has_zero && index_of_negative_1 != -1) {
        std::string error_msg = "cannot reshape tensor of 0 elements into shape (";
        for (const auto& s : shape) {
            error_msg += std::to_string(s) + ",";
        }
        error_msg += ") because the unspecified dimension size -1 can be any value and is ambiguous";
        TT_THROW("{}", error_msg);
    }

    ttsl::SmallVector<uint32_t> new_shape(shape.size());
    std::copy(shape.begin(), shape.end(), new_shape.begin());
    if (index_of_negative_1 == -1) {
        TT_FATAL(new_volume == old_volume, "Invalid arguments to reshape");
    } else {
        TT_FATAL(old_volume % new_volume == 0, "Invalid arguments to reshape");
        new_shape[index_of_negative_1] = old_volume / new_volume;
    }

    return tt::tt_metal::Shape(std::move(new_shape));
}

int compute_flat_indices(tt::stl::Span<const int> indices, tt::stl::Span<const uint64_t> strides) {
    int flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

std::size_t compute_buffer_size(const tt::tt_metal::Shape& shape, DataType data_type, const Tile& tile) {
    const size_t volume = shape.volume();
    auto tile_hw = tile.get_tile_hw();
    if (data_type == DataType::BFLOAT8_B) {
        auto tile_size_bytes = tile.get_tile_size(DataFormat::Bfp8_b);
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat8_b_volume = volume / tile_hw * tile_size_bytes;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat8_b_volume / sizeof(std::uint32_t);
    }
    if (data_type == DataType::BFLOAT4_B) {
        auto tile_size_bytes = tile.get_tile_size(DataFormat::Bfp4_b);
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat4_b_volume = volume / tile_hw * tile_size_bytes;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat4_b_volume / sizeof(std::uint32_t);
    }
    return volume;
}

bool is_arch_gs(const tt::ARCH& arch) { return arch == tt::ARCH::GRAYSKULL; }

bool is_arch_whb0(const tt::ARCH& arch) { return arch == tt::ARCH::WORMHOLE_B0; }

bool is_cpu_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::HOST; }

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape) {
    const auto num_shards_height = tt::div_up(shape.height(), shard_shape.height());
    const auto last_shard_height =
        shape.height() % shard_shape.height() > 0 ? shape.height() % shard_shape.height() : shard_shape.height();
    const auto num_shards_width = tt::div_up(shape.width(), shard_shape.width());
    const auto last_shard_width =
        shape.width() % shard_shape.width() > 0 ? shape.width() % shard_shape.width() : shard_shape.width();

    return ShardDivisionSpec{num_shards_height, last_shard_height, num_shards_width, last_shard_width};
};

MemoryConfig compute_auto_shard_spec(const Tensor& input_tensor, const MemoryConfig& output_memory_config) {
    // If output memory config is not sharded or already has a shard_spec, return as-is
    if (!output_memory_config.is_sharded() || output_memory_config.shard_spec().has_value()) {
        return output_memory_config;
    }

    // If input tensor has a shard_spec, reuse it
    if (input_tensor.is_sharded() && input_tensor.shard_spec().has_value()) {
        return output_memory_config.with_shard_spec(input_tensor.shard_spec().value());
    }

    // Create a default shard_spec based on input shape and memory layout
    const auto& input_shape = input_tensor.padded_shape();
    const auto& device = input_tensor.device();
    const auto memory_layout = output_memory_config.memory_layout();

    // Get device compute grid
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    // Calculate shard shape based on memory layout
    // Shard dimensions must satisfy:
    // 1. Tile alignment: multiples of tile size (32x32)
    // 2. L1 alignment: (shard_width * datum_size) % L1_alignment == 0
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_HEIGHT = 32;

    // Get L1 alignment requirement and datum size for alignment calculation
    uint32_t l1_alignment = hal::get_l1_alignment();
    DataType input_dtype = input_tensor.dtype();
    tt::DataFormat input_df = tt::tt_metal::datatype_to_dataformat_converter(input_dtype);
    uint32_t datum_size_bytes = datum_size(input_df);
    // Calculate minimum shard width in elements to satisfy L1 alignment
    // L1 alignment is in bytes, so we need: shard_width_elements * datum_size_bytes >= l1_alignment
    // and shard_width_elements * datum_size_bytes % l1_alignment == 0
    uint32_t min_shard_width_for_l1 = tt::div_up(l1_alignment, datum_size_bytes);
    // Round up to tile boundary for L1-aligned shard width
    uint32_t l1_aligned_tile_multiple = tt::round_up(min_shard_width_for_l1, TILE_WIDTH);

    std::array<uint32_t, 2> shard_shape;
    uint32_t num_cores = 0;
    uint32_t total_height = input_tensor.physical_volume() / input_shape[-1];
    uint32_t total_width = input_shape[-1];

    switch (memory_layout) {
        case TensorMemoryLayout::WIDTH_SHARDED: {
            // For width sharding, maximize parallelism while ensuring alignment
            TT_FATAL(
                total_width >= l1_aligned_tile_multiple,
                "Invalid configuration for WIDTH_SHARDED: total_width ({}) must be >= l1_aligned_tile_multiple ({})",
                total_width,
                l1_aligned_tile_multiple);

            uint32_t max_grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
            // Try to maximize core usage: start with max_grid_cores and find the smallest shard_width
            // that satisfies alignment and allows using as many cores as possible
            uint32_t max_possible_cores = std::min(max_grid_cores, total_width / l1_aligned_tile_multiple);
            max_possible_cores = std::max(1u, max_possible_cores);

            // Find the maximum number of cores we can actually use by trying different shard widths
            // Start from max_possible_cores and work backwards to find the maximum that works
            uint32_t best_num_cores = 1;
            uint32_t best_shard_width = total_width;

            for (uint32_t try_cores = max_possible_cores; try_cores >= 1; --try_cores) {
                uint32_t shard_width = tt::round_up(total_width / try_cores, l1_aligned_tile_multiple);
                shard_width = std::max(shard_width, l1_aligned_tile_multiple);
                shard_width = std::min(shard_width, total_width);

                uint32_t actual_cores = tt::div_up(total_width, shard_width);
                if (actual_cores <= max_grid_cores && actual_cores >= best_num_cores) {
                    best_num_cores = actual_cores;
                    best_shard_width = shard_width;
                    // If we found a solution that uses max_grid_cores, we're done
                    if (best_num_cores >= max_possible_cores) {
                        break;
                    }
                }
            }

            shard_shape = {total_height, best_shard_width};
            num_cores = best_num_cores;
            break;
        }
        case TensorMemoryLayout::HEIGHT_SHARDED: {
            // For height sharding, maximize parallelism by dividing height across available cores
            TT_FATAL(
                total_height >= TILE_HEIGHT,
                "Invalid configuration for HEIGHT_SHARDED: total_height ({}) must be >= TILE_HEIGHT ({})",
                total_height,
                TILE_HEIGHT);

            uint32_t max_grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
            // Try to maximize core usage: start with max_grid_cores and find the smallest shard_height
            // that satisfies tile alignment and allows using as many cores as possible
            uint32_t max_possible_cores = std::min(max_grid_cores, total_height / TILE_HEIGHT);
            max_possible_cores = std::max(1u, max_possible_cores);

            // Find the maximum number of cores we can actually use by trying different shard heights
            // Start from max_possible_cores and work backwards to find the maximum that works
            uint32_t best_num_cores = 1;
            uint32_t best_shard_height = total_height;

            for (uint32_t try_cores = max_possible_cores; try_cores >= 1; --try_cores) {
                uint32_t shard_height = tt::round_up(total_height / try_cores, TILE_HEIGHT);
                shard_height = std::max(shard_height, TILE_HEIGHT);
                shard_height = std::min(shard_height, total_height);

                uint32_t actual_cores = tt::div_up(total_height, shard_height);
                if (actual_cores <= max_grid_cores && actual_cores >= best_num_cores) {
                    best_num_cores = actual_cores;
                    best_shard_height = shard_height;
                    // If we found a solution that uses max_possible_cores, we're done
                    if (best_num_cores >= max_possible_cores) {
                        break;
                    }
                }
            }

            shard_shape = {best_shard_height, total_width};
            num_cores = best_num_cores;
            break;
        }
        case TensorMemoryLayout::BLOCK_SHARDED: {
            // For block sharding, maximize core usage by dividing both dimensions across grid
            // For COL_MAJOR orientation: num_shards_along_width <= grid.y, num_shards_along_height <= grid.x
            uint32_t grid_y = static_cast<uint32_t>(grid_size.y);  // rows
            uint32_t grid_x = static_cast<uint32_t>(grid_size.x);  // columns
            uint32_t max_grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);

            // Try to maximize core usage by finding the best combination of num_shards_height and num_shards_width
            // that satisfies constraints and uses as many cores as possible
            uint32_t best_num_cores = 0;
            uint32_t best_shard_height = 0;
            uint32_t best_shard_width = 0;

            // Try different combinations of num_shards_height and num_shards_width
            // Start with maximum possible and work down to find the best fit
            for (uint32_t try_num_shards_h = std::min(grid_x, tt::div_up(total_height, TILE_HEIGHT));
                 try_num_shards_h >= 1;
                 --try_num_shards_h) {
                // Calculate shard height for this number of shards
                uint32_t min_shard_height = tt::div_up(total_height, try_num_shards_h);
                uint32_t shard_height = tt::round_up(min_shard_height, TILE_HEIGHT);
                shard_height = std::max(shard_height, TILE_HEIGHT);
                shard_height = std::min(shard_height, total_height);
                uint32_t actual_num_shards_h = tt::div_up(total_height, shard_height);

                // If rounding caused violation, skip this combination
                if (actual_num_shards_h > grid_x) {
                    continue;
                }

                // Now try different widths for this height
                for (uint32_t try_num_shards_w = std::min(grid_y, tt::div_up(total_width, l1_aligned_tile_multiple));
                     try_num_shards_w >= 1;
                     --try_num_shards_w) {
                    // Calculate shard width for this number of shards
                    uint32_t min_shard_width = tt::div_up(total_width, try_num_shards_w);
                    uint32_t shard_width = tt::round_up(min_shard_width, l1_aligned_tile_multiple);
                    shard_width = std::max(shard_width, l1_aligned_tile_multiple);
                    shard_width = std::min(shard_width, total_width);
                    uint32_t actual_num_shards_w = tt::div_up(total_width, shard_width);

                    // If rounding caused violation, skip this combination
                    if (actual_num_shards_w > grid_y) {
                        continue;
                    }

                    uint32_t total_cores = actual_num_shards_h * actual_num_shards_w;
                    if (total_cores <= max_grid_cores && total_cores > best_num_cores) {
                        best_num_cores = total_cores;
                        best_shard_height = shard_height;
                        best_shard_width = shard_width;
                        // If we found a solution that uses max_grid_cores, we're done
                        if (best_num_cores >= max_grid_cores) {
                            goto found_best_block_sharding;
                        }
                    }
                }
            }

        found_best_block_sharding:
            TT_FATAL(best_num_cores > 0, "Could not find valid block sharding configuration");

            shard_shape = {best_shard_height, best_shard_width};
            num_cores = best_num_cores;
            break;
        }
        default: TT_FATAL(false, "Unsupported sharding scheme for automatic shard_spec creation");
    }

    // Create core range set
    CoreRangeSet grid_set;
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // For BLOCK_SHARDED, create a rectangular grid directly
        // For COL_MAJOR orientation: num_shards_height maps to grid.x (columns), num_shards_width maps to grid.y (rows)
        // CoreRange uses (x, y) coordinates where x=columns, y=rows
        uint32_t num_shards_h = tt::div_up(total_height, shard_shape[0]);  // num_shards_height
        uint32_t num_shards_w = tt::div_up(total_width, shard_shape[1]);   // num_shards_width
        // TT_FATAL assertions above guarantee num_shards_h <= grid_size.x and num_shards_w <= grid_size.y
        // Create a single rectangular CoreRange: (x=num_shards_height, y=num_shards_width)
        CoreRange block_range({0, 0}, {num_shards_h - 1, num_shards_w - 1});
        grid_set = CoreRangeSet({block_range});
    } else {
        bool row_wise = (memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
        grid_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, row_wise);
    }

    // Determine shard orientation
    ShardOrientation shard_orientation = (memory_layout == TensorMemoryLayout::WIDTH_SHARDED)
                                             ? ShardOrientation::ROW_MAJOR
                                             : ShardOrientation::COL_MAJOR;

    // Create shard spec
    ShardSpec shard_spec(grid_set, shard_shape, shard_orientation);
    return output_memory_config.with_shard_spec(shard_spec);
}

}  // namespace tt_metal

}  // namespace tt
