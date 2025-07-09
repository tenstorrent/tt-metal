// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

ttnn::Shape squeeze_shape_to_ND(const ttnn::Shape& shape, const uint32_t n) {
    if (shape.rank() <= n) {
        return shape;
    }
    ttnn::SmallVector<uint32_t> shape_nd(n);
    std::copy(shape.view().rbegin(), shape.view().rbegin() + n, shape_nd.rbegin());
    const auto rank_diff_end = shape.rank() - n + 1;
    shape_nd[0] = std::accumulate(shape.cbegin(), shape.cbegin() + rank_diff_end, 1, std::multiplies<uint32_t>());

    return ttnn::Shape(shape_nd);
}

ttnn::Shape squeeze_shape_to_4D(const ttnn::Shape& shape) { return squeeze_shape_to_ND(shape, 4); }
ttnn::Shape squeeze_shape_to_3D(const ttnn::Shape& shape) { return squeeze_shape_to_ND(shape, 3); }


ttnn::Tensor squeeze_from_ND_to_4D(const ttnn::Tensor& tensor) {
    auto shape = tensor.logical_shape();
    auto rank = shape.rank();
    TT_FATAL(shape.rank() >= 4, "Tensor has to be of rank larger than 4! Instead is {}", shape.rank());
    if (rank == 4) {
        return tensor;
    }
    int i = 0;
    // This is a workaround for now, it will be fixed in another PR
    if (shape[i] == 1) {
        auto squeezed = tensor;
        while (rank > 4 && shape[i] == 1) {
            squeezed = ttnn::squeeze(squeezed, 0);
            rank = squeezed.logical_shape().rank();
            i++;
        }
        if (rank <= 4) {
            return squeezed;
        }
        return ttnn::reshape(squeezed, squeeze_shape_to_4D(shape));
    }
    return ttnn::reshape(tensor, squeeze_shape_to_4D(shape));
}

ttnn::Shape unsqueeze_shape_to_ND(const ttnn::Shape& shape, const uint32_t n) {
    ttnn::SmallVector<uint32_t> shape_vector(n, 1);
    std::copy(shape.view().rbegin(), shape.view().rend(), shape_vector.rbegin());
    return ttnn::Shape(shape_vector);
}


ttnn::Shape unsqueeze_shape_to_3D(const ttnn::Shape& shape) { return unsqueeze_shape_to_ND(shape, 3); };
ttnn::Shape unsqueeze_shape_to_4D(const ttnn::Shape& shape) { return unsqueeze_shape_to_ND(shape, 4); };


ttnn::Shape squeeze_or_unsqueeze_shape_to_ND(const ttnn::Shape& shape, const uint32_t n) {
    const auto input_rank = shape.rank();
    if (input_rank == n) {
        return shape;
    } else if (input_rank < n) {
        return unsqueeze_shape_to_ND(shape, n);
    } else {
        return squeeze_shape_to_ND(shape, n);
    }
}


uint32_t get_estimated_size_of_cbs(
    const Tensor& input_tensor_a,
    const uint32_t input_single_tile_size,
    const uint32_t output_single_tile_size,
    const uint32_t num_tiles_per_row) {
    uint32_t cb_src0_size = input_single_tile_size * num_tiles_per_row;
    uint32_t cb_output_size = output_single_tile_size * num_tiles_per_row;
    return cb_src0_size + cb_output_size;
}

uint32_t get_max_l1_space(const Tensor& input_tensor_a) {
    auto device = input_tensor_a.device();
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return max_l1_space;
}

bool is_enough_space(
    const Tensor& input_tensor_a,
    const uint32_t input_single_tile_size,
    const uint32_t output_single_tile_size,
    const uint32_t num_tiles_per_row) {
    uint32_t max_l1_space = get_max_l1_space(input_tensor_a);
    uint32_t estimated_size_of_cbs =
        get_estimated_size_of_cbs(input_tensor_a, input_single_tile_size, output_single_tile_size, num_tiles_per_row);
    return max_l1_space > estimated_size_of_cbs;
}

ttnn::Tensor pad_to_tile_vol(
    QueueId queue_id,
    const ttnn::Tensor& tensor,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& logical_shape = tensor.logical_shape();
    auto padded_shape = tensor.padded_shape();
    auto rank = logical_shape.rank();
    if (padded_shape[-1] % tt::constants::TILE_WIDTH != 0 || padded_shape[-2] % tt::constants::TILE_HEIGHT != 0) {
        TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

        auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
        auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
        uint32_t num_non_hw_dims = rank - 2u;
        auto padding_vec = ttnn::SmallVector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0, 0});
        padding_vec.reserve(rank);
        padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
        padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

        constexpr bool pad_use_multicore = true;
        auto padded_output = ttnn::pad(queue_id, tensor, padding_vec, value, use_multicore, memory_config);
        TT_FATAL(
            padded_output.padded_shape()[-1] % tt::constants::TILE_WIDTH == 0 &&
                padded_output.padded_shape()[-2] % tt::constants::TILE_HEIGHT == 0,
            "pad_to_tile_vol: output tensor must be divisible by tile size");
        return padded_output;
    }
    return tensor;
}
uint32_t wrap_index(int index, int size) { return index < 0 ? size + index : index; }

ttnn::Shape compute_padded_shape(
    const ttnn::Shape& logical_shape, const uint32_t tile_height, const uint32_t tile_width) {
    if (logical_shape.rank() == 1) {
        return ttnn::Shape{tile_height, tile_width};
    }

    ttnn::SmallVector<uint32_t> output_shape_vec(logical_shape.rank());
    std::copy(logical_shape.cbegin(), logical_shape.cend(), output_shape_vec.begin());

    const std::array<uint32_t, 2> tile_shape = {tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT};
    auto shapeit = tile_shape.rbegin();

    std::for_each(output_shape_vec.rbegin(), output_shape_vec.rbegin() + 2, [&shapeit](auto& x) {
        x = tt::round_up(x, *(shapeit++));
    });

    return ttnn::Shape(output_shape_vec);
}

std::array<uint32_t, 2> compute_block_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                          const tt::tt_metal::Layout& layout,
                                                          const tt::tt_metal::CoreCoord& grid_size,
                                                          const tt::tt_metal::ShardOrientation& orientation,
                                                          const uint32_t total_num_cores) {
    TT_FATAL(grid_size.y * grid_size.x == total_num_cores, "compute_block_sharded_shard_shape received a core grid shape that does not match the total number of cores");
    auto adjusted_grid_size = grid_size;
    if (orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
        // for col major, we partition the width of the tensor along the height of the core grid
        std::swap(adjusted_grid_size.x, adjusted_grid_size.y);
    }

    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto tensor_height_padded_to_tile =
        layout == tt::tt_metal::Layout::TILE
            ? tt::round_up(tensor_height, adjusted_grid_size.y * tt::constants::TILE_HEIGHT)
            : tensor_height;
    std::array<uint32_t, 2> shard_shape = {tt::div_up(tensor_height_padded_to_tile, adjusted_grid_size.y),
                                           tt::div_up(tensor_width, adjusted_grid_size.x)};

    return shard_shape;
}

std::array<uint32_t, 2> compute_width_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                          const uint32_t total_num_cores) {
    return {squeezed_tensor_hw[0], tt::div_up(squeezed_tensor_hw[1], total_num_cores)};
}

std::array<uint32_t, 2> compute_height_sharded_shard_shape(const std::array<uint32_t, 2>& squeezed_tensor_hw,
                                                           const tt::tt_metal::Layout& layout,
                                                           const uint32_t total_num_cores) {
    auto [tensor_height, tensor_width] = squeezed_tensor_hw;
    auto squeezed_height_padded_to_tile = layout == tt::tt_metal::Layout::TILE
                                                    ? tt::round_up(tensor_height, total_num_cores)
                                                    : tensor_height;
    return {tt::div_up(squeezed_height_padded_to_tile, total_num_cores), tensor_width};
}

ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    std::optional<std::array<uint32_t, 2>> shard_shape,
    const tt::tt_metal::Layout& layout) {
    auto rank = logical_shape.rank();
    TT_FATAL(rank >= 2, "rank of tensor to shard must be at least 2.");

    ttnn::TensorMemoryLayout tensor_memory_layout{};
    if (strategy == ShardStrategy::BLOCK) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    } else if (strategy == ShardStrategy::WIDTH) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::WIDTH_SHARDED;
    } else if (strategy == ShardStrategy::HEIGHT) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
    }

    auto height = logical_shape[-2];
    auto width = logical_shape[-1];
    std::array<uint32_t, 2> computed_shard_shape;

    if (shard_shape.has_value()) {
        computed_shard_shape = shard_shape.value();
    } else {
        uint32_t batch_size = 1;
        for (int i = 0; i < rank - 2; i++) {
            batch_size *= logical_shape[i];
        }

        auto tensor_height = batch_size * height;
        auto tensor_width = width;
        std::array<uint32_t, 2> squeezed_tensor_hw{tensor_height, tensor_width};
        auto total_num_cores = core_grid.num_cores();
        CoreCoord grid_size = core_grid.bounding_box().grid_size();

        switch (strategy) {
            case ShardStrategy::BLOCK:
                computed_shard_shape = compute_block_sharded_shard_shape(squeezed_tensor_hw, layout, grid_size, orientation, total_num_cores);
                break;
            case ShardStrategy::WIDTH:
                computed_shard_shape = compute_width_sharded_shard_shape(squeezed_tensor_hw, total_num_cores);
                break;
            case ShardStrategy::HEIGHT:
                computed_shard_shape = compute_height_sharded_shard_shape(squeezed_tensor_hw, layout, total_num_cores);
                break;
            default:
                TT_ASSERT(false, "Invalid shard strategy");
        }
    }

    if (layout == tt::tt_metal::Layout::TILE) {
        auto [shard_height, shard_width] = computed_shard_shape;
        auto tile_divides_shard_height = shard_height % tt::constants::TILE_HEIGHT == 0;
        auto tile_divides_shard_width = shard_width % tt::constants::TILE_WIDTH == 0;
        TT_FATAL(tile_divides_shard_width && tile_divides_shard_height,
                 "For sharding tiled tensors, the shard shape must fit neatly into tiles but "
                 "create_sharded_memory_config got shard width {} and shard height {} while "
                 "on this architecture we have tile width {} and tile height {}",
                 computed_shard_shape[0], computed_shard_shape[1], tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT);
    }

    auto shard_spec = tt::tt_metal::ShardSpec(core_grid, computed_shard_shape, orientation);
    return ttnn::MemoryConfig(tensor_memory_layout, ttnn::BufferType::L1, shard_spec);
}

std::pair<uint32_t, std::array<uint32_t, 2>> tensor_coord_to_height_sharded_coord(
    const std::span<const uint32_t>& tensor_shape,
    const std::span<const uint32_t>& shard_shape,
    const std::span<const uint32_t>& tensor_coord) {
    std::array<uint32_t, 2> tensor_shape_2d{0, 0};
    for (size_t i = 0; i < tensor_shape.size(); i++) {
        if (i == tensor_shape.size() - 1) {
            // width dimension, goes unmodified
            tensor_shape_2d[1] = tensor_shape[i];
        } else {
            // height dimension, squeeze into 2D shape
            if (tensor_shape_2d[0] == 0) {
                // first time we've seen this dimension
                tensor_shape_2d[0] = tensor_shape[i];
            } else {
                tensor_shape_2d[0] *= tensor_shape[i];
            }
        }
    }

    std::array<uint32_t, 2> tensor_coord_2d{0, tensor_coord.back()};
    uint32_t height_2d = 0;
    for (size_t i = 0; i < tensor_coord.size() - 1; i++) {
        std::vector<uint32_t> page_shapes(tensor_shape.begin() + i + 1, tensor_shape.end() - 1);
        auto component_sum =
            tensor_coord[i] * std::accumulate(page_shapes.begin(), page_shapes.end(), 1, std::multiplies<uint32_t>());
        height_2d += component_sum;
    }
    tensor_coord_2d[0] = height_2d;

    uint32_t shard_height = shard_shape[0];
    uint32_t w_in_shard = tensor_coord_2d[1];
    uint32_t h_in_shard = height_2d % shard_height;
    uint32_t which_shard = height_2d / shard_height;

    std::array<uint32_t, 2> shard_coord{h_in_shard, w_in_shard};
    return std::make_pair(which_shard, shard_coord);
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
