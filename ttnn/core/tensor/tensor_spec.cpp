// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void validate_shard_spec_with_tensor_shape(const TensorSpec& tensor_spec) {
    const auto& memory_config = tensor_spec.memory_config();
    if (!memory_config.is_sharded() or !memory_config.shard_spec().has_value()) {
        return;
    }
    // Sharding checks use physical shape and physical shard shape
    // TODO: Review and port to use logical shapes
    const auto& physical_shape = tensor_spec.physical_shape();
    const auto physical_height = physical_shape.height();
    const auto physical_width = physical_shape.width();

    const auto& physical_shard_shape = tensor_spec.tensor_layout().get_physical_shard_shape();
    const auto physical_shard_height = physical_shard_shape.height();
    const auto physical_shard_width = physical_shard_shape.width();

    const auto& shard_spec = memory_config.shard_spec().value();
    uint32_t num_cores = shard_spec.num_cores();

    // TODO (issue #17060): Flip to TT_FATAL
    if (memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(
            physical_width == physical_shard_width,
            "Shard width {} must match physical width {} for height sharded",
            physical_shard_width,
            physical_width);
        uint32_t num_shards = div_up(physical_height, physical_shard_height);
        TT_FATAL(
            num_shards <= num_cores,
            "Number of shards along height {} must not exceed number of cores {}",
            num_shards,
            num_cores);
    } else if (memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_FATAL(
            physical_height == physical_shard_height,
            "Shard height {} must match physical height {} for width sharded",
            physical_shard_height,
            physical_height);
        uint32_t num_shards = div_up(physical_width, physical_shard_width);
        TT_FATAL(
            num_shards <= num_cores,
            "Number of shards along width {} must not exceed number of cores {}",
            num_shards,
            num_cores);
    } else if (memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            shard_spec.grid.ranges().size() == 1, "Shard grid must be one full rectangular grid for block sharded!");
        uint32_t num_shards_along_height = div_up(physical_height, physical_shard_height);
        uint32_t num_shards_along_width = div_up(physical_width, physical_shard_width);

        // Additionally check that number of cores along height and width matches shard grid
        const CoreCoord shard_grid = shard_spec.grid.bounding_box().grid_size();
        if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
            TT_FATAL(
                num_shards_along_height <= shard_grid.y,
                "Number of shards along height {} must not exceed number of rows {} for row major orientation!",
                num_shards_along_height,
                shard_grid.y);
            TT_FATAL(
                num_shards_along_width <= shard_grid.x,
                "Number of shards along width {} must not exceed number of columns {} for row major orientation!",
                num_shards_along_width,
                shard_grid.x);
        } else {
            TT_FATAL(
                num_shards_along_height <= shard_grid.x,
                "Number of shards along height {} must not exceed number of columns {} for column major "
                "orientation!",
                num_shards_along_height,
                shard_grid.x);
            TT_FATAL(
                num_shards_along_width <= shard_grid.y,
                "Number of shards along width {} must not exceed number of rows {} for column major orientation!",
                num_shards_along_width,
                shard_grid.y);
        }
    }
}

void validate_dtype_and_layout(DataType dtype, Layout layout) {
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (dtype == DataType::UINT32 || dtype == DataType::INT32 || dtype == DataType::FLOAT32 ||
             dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::BFLOAT16 ||
             dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B),
            "Only UINT32, INT32, FLOAT32, UINT16, UINT8, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on "
            "device!");
    };
    auto supported_layout = [&dtype, &layout]() {
        switch (dtype) {
            case DataType::UINT32:
            case DataType::INT32:
            case DataType::FLOAT32:
            case DataType::UINT8:
            case DataType::UINT16:
            case DataType::BFLOAT16: break;
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
                TT_ASSERT(layout == Layout::TILE, "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(
                    false,
                    "Only UINT32, INT32, FLOAT32, UINT16, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on "
                    "device!");
                break;
        }
    };
    supported_dtype();
    supported_layout();
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TensorSpec::TensorSpec(ttnn::Shape logical_shape, TensorLayout tensor_layout) :
    logical_shape_(std::move(logical_shape)),
    tensor_layout_(std::move(tensor_layout)),
    cached_padded_shape_(tensor_layout_.compute_padded_shape(logical_shape_)),
    cached_logical_2d_shape_(tensor_layout_.compute_logical_2d_shape(logical_shape_)),
    cached_physical_shape_(tensor_layout_.compute_physical_shape(logical_shape_)) {
    CMAKE_UNIQUE_NAMESPACE::validate_shard_spec_with_tensor_shape(*this);
    CMAKE_UNIQUE_NAMESPACE::validate_dtype_and_layout(data_type(), layout());
    populate_sharding_specs();
}

TensorSpec TensorSpec::with_memory_config(MemoryConfig memory_config) const {
    TensorSpec result = *this;
    result.tensor_layout_ = tensor_layout_.with_memory_config(std::move(memory_config));
    result.populate_sharding_specs();
    return result;
}

TensorSpec TensorSpec::sharded_across_dims(
    tt::stl::Span<const int32_t> dims, CoreRangeSet grid, ShardOrientation orientation) const {
    Shape shard_shape = padded_shape();
    for (auto dim : dims) {
        shard_shape[dim] = 1;
    }
    NdShardSpec shard_spec(std::move(shard_shape), std::move(grid), orientation);
    return sharded(std::move(shard_spec), ShardShapeAlignment::RECOMMENDED);
}

TensorSpec TensorSpec::sharded_across_dims_except(
    tt::stl::Span<const int32_t> dims, CoreRangeSet grid, ShardOrientation orientation) const {
    const auto& padded_shape = this->padded_shape();
    Shape shard_shape = Shape().to_rank(padded_shape.rank());
    for (auto dim : dims) {
        shard_shape[dim] = padded_shape[dim];
    }
    auto shard_spec = NdShardSpec(std::move(shard_shape), std::move(grid), orientation);
    return sharded(std::move(shard_spec), ShardShapeAlignment::RECOMMENDED);
}

TensorSpec TensorSpec::height_sharded(CoreRangeSet grid, ShardOrientation orientation) const {
    auto num_cores = grid.num_cores();
    auto shard_height = div_up(physical_shape().height(), num_cores);
    NdShardSpec shard_spec(Shape({shard_height, physical_shape().width()}), std::move(grid), orientation);
    return sharded(std::move(shard_spec), ShardShapeAlignment::REQUIRED);
}

TensorSpec TensorSpec::width_sharded(CoreRangeSet grid, ShardOrientation orientation) const {
    auto num_cores = grid.num_cores();
    auto shard_width = div_up(physical_shape().width(), num_cores);
    NdShardSpec shard_spec(Shape({physical_shape().height(), shard_width}), std::move(grid), orientation);
    return sharded(std::move(shard_spec), ShardShapeAlignment::REQUIRED);
}

TensorSpec TensorSpec::block_sharded(CoreRange grid, ShardOrientation orientation) const {
    auto grid_size = grid.grid_size();
    auto shard_height =
        div_up(physical_shape().height(), orientation == ShardOrientation::ROW_MAJOR ? grid_size.y : grid_size.x);
    auto shard_width =
        div_up(physical_shape().width(), orientation == ShardOrientation::ROW_MAJOR ? grid_size.x : grid_size.y);
    NdShardSpec shard_spec(Shape({shard_height, shard_width}), std::move(grid), orientation);
    return sharded(std::move(shard_spec), ShardShapeAlignment::RECOMMENDED);
}

TensorSpec TensorSpec::sharded(NdShardSpec nd_shard_spec, ShardShapeAlignment shard_alignment) const {
    if (shard_alignment != ShardShapeAlignment::NONE) {
        auto alignment = shard_alignment == ShardShapeAlignment::REQUIRED
                             ? page_config().get_required_shard_shape_alignment()
                             : page_config().get_recommended_shard_shape_alignment(data_type());
        auto& shard_shape = nd_shard_spec.shard_shape;
        for (int dim = 1; dim <= alignment.size(); dim++) {
            shard_shape[-dim] = round_up(shard_shape[-dim], alignment[-dim]);
        }
    }
    TensorLayout new_layout(
        data_type(), page_config(), MemoryConfig(memory_config().buffer_type(), std::move(nd_shard_spec)));
    return TensorSpec(logical_shape(), std::move(new_layout));
}

TensorSpec TensorSpec::sharded(
    Shape shard_shape,
    CoreRangeSet grid,
    ShardShapeAlignment shard_alignment,
    ShardOrientation orientation,
    ShardDistributionStrategy shard_distribution_strategy) const {
    return sharded(
        NdShardSpec(std::move(shard_shape), std::move(grid), orientation, shard_distribution_strategy),
        shard_alignment);
}

void TensorSpec::populate_sharding_specs() {
    if (memory_config().created_with_nd_shard_spec()) {
        if (auto upd_mem_config = populate_legacy_shard_spec_from_nd()) {
            tensor_layout_ = tensor_layout_.with_memory_config(std::move(*upd_mem_config));
        }
    } else if (memory_config().shard_spec() && memory_config().shard_spec()->mode == ShardMode::PHYSICAL) {
        tensor_layout_ = tensor_layout_.with_memory_config(populate_nd_shard_spec_from_legacy());
    }
}

MemoryConfig TensorSpec::populate_nd_shard_spec_from_legacy() const {
    const auto& mem_config = memory_config();
    auto mem_layout = mem_config.memory_layout();
    const auto& shard_spec = mem_config.shard_spec().value();

    NdShardSpec nd_shard_spec{
        .shard_shape = Shape({shard_spec.shape[0], shard_spec.shape[1]}),
        .grid = shard_spec.grid,
        .orientation = shard_spec.orientation,
    };
    if (padded_shape().rank() == 1) {
        TT_FATAL(shard_spec.shape[0] == 1, "Shard shape must be 1D for 1D tensor!");
        nd_shard_spec.shard_shape = Shape({shard_spec.shape[1]});
    }

    // For block sharding, we need to use 2D grid distribution to ensure the same distribution of shards
    if (mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        nd_shard_spec.shard_distribution_strategy = ShardDistributionStrategy::GRID_2D;
    }

    return MemoryConfig::create_with_prepopulated_shard_specs(
        mem_config.memory_layout(),
        mem_config.buffer_type(),
        mem_config.shard_spec(),
        std::move(nd_shard_spec),
        mem_config.created_with_nd_shard_spec());
}

std::optional<MemoryConfig> TensorSpec::populate_legacy_shard_spec_from_nd() const {
    const auto& mem_config = memory_config();
    const auto& nd_shard_spec = mem_config.nd_shard_spec().value();
    const auto& nd_shard_shape = nd_shard_spec.shard_shape;

    // Trying to flatten ND shard shape into 2D
    std::array<uint32_t, 2> shard_shape = {1, nd_shard_shape[-1]};
    size_t cur_tensor_volume = padded_shape()[-1];
    for (int dim = -2; dim >= -static_cast<int>(nd_shard_shape.rank()); dim--) {
        auto tensor_size = padded_shape()[dim];
        auto shard_size = nd_shard_shape[dim];
        cur_tensor_volume *= tensor_size;
        shard_shape[0] *= shard_size;

        // Folding dimensions if shard shape is identical to tensor shape
        if (tensor_size == shard_size) {
            continue;
        }

        // All folding is now complete, determining if the calculated shard spec matches the original one

        // For conversion to be possible, this dimension must be the highest non-1 dimension in the shard shape
        if (nd_shard_shape.volume() != shard_shape[0] * shard_shape[1]) {
            return std::nullopt;
        }

        // We can perform the conversion in one of 3 cases:
        // 1. The current tensor dimension is divisible by the shard dimension
        // 2. This is the last dimension of the shard shape (not even 1s are allowed)
        // 3. All higher tensor dimensions are also 1s
        bool is_last_dim = dim == -static_cast<int>(nd_shard_shape.rank());
        bool is_divisible = tensor_size % shard_size == 0;
        bool all_next_tensor_dims_are_ones = padded_shape().volume() == cur_tensor_volume;
        if (is_last_dim || is_divisible || all_next_tensor_dims_are_ones) {
            break;
        }

        return std::nullopt;
    }

    ShardSpec shard_spec(nd_shard_spec.grid, shard_shape, nd_shard_spec.orientation);

    // Check that the number of shards fits onto the cores
    size_t num_shards_along_height = div_up(physical_shape().height(), shard_spec.shape[0]);
    size_t num_shards_along_width = div_up(physical_shape().width(), shard_spec.shape[1]);
    if (shard_spec.orientation != ShardOrientation::ROW_MAJOR) {
        std::swap(num_shards_along_height, num_shards_along_width);
    }
    size_t total_num_shards = num_shards_along_height * num_shards_along_width;
    if (total_num_shards > shard_spec.grid.num_cores()) {
        return std::nullopt;
    }

    TensorMemoryLayout shard_kind = TensorMemoryLayout::BLOCK_SHARDED;
    if (nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D) {
        if (shard_spec.shape[0] == padded_shape().volume() / padded_shape()[-1]) {
            shard_kind = TensorMemoryLayout::WIDTH_SHARDED;
        } else if (shard_spec.shape[1] == padded_shape()[-1]) {
            shard_kind = TensorMemoryLayout::HEIGHT_SHARDED;
        }
    }

    if (shard_kind != TensorMemoryLayout::BLOCK_SHARDED) {
        return MemoryConfig::create_with_prepopulated_shard_specs(
            shard_kind,
            mem_config.buffer_type(),
            std::move(shard_spec),
            mem_config.nd_shard_spec(),
            mem_config.created_with_nd_shard_spec());
    }

    // Block sharding requires a contiguous grid of cores
    if (shard_spec.grid.ranges().size() != 1) {
        return std::nullopt;
    }

    // If 1D distribution is used, we need the number of shards along width to match the grid width to guarantee the
    // same distribution of shards
    CoreCoord shard_grid = shard_spec.grid.ranges()[0].grid_size();
    if (nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D &&
        num_shards_along_width != shard_grid.x) {
        return std::nullopt;
    }

    // To match the shard distribution, the number of shards along width and height must fit into the grid
    if (num_shards_along_width > shard_grid.x || num_shards_along_height > shard_grid.y) {
        return std::nullopt;
    }

    return MemoryConfig::create_with_prepopulated_shard_specs(
        TensorMemoryLayout::BLOCK_SHARDED,
        mem_config.buffer_type(),
        std::move(shard_spec),
        mem_config.nd_shard_spec(),
        mem_config.created_with_nd_shard_spec());
}

}  // namespace tt::tt_metal
