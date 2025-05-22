// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_spec.hpp"

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

void TensorSpec::populate_sharding_specs() {
    if (!memory_config().nd_shard_spec().has_value()) {
        if (auto nd_shard_spec = nd_shard_spec_from_legacy()) {
            auto mem_config = memory_config();
            mem_config.nd_shard_spec_ = std::move(nd_shard_spec);
            tensor_layout_ = tensor_layout_.with_memory_config(std::move(mem_config));
        }
    }
    if (!memory_config().shard_spec().has_value() && memory_config().nd_shard_spec().has_value()) {
        if (auto shard_spec = legacy_shard_spec_from_nd()) {
            auto mem_config = memory_config();
            mem_config.shard_spec_ = std::move(shard_spec);
            tensor_layout_ = tensor_layout_.with_memory_config(std::move(mem_config));
        }
    }
}

std::optional<NdShardSpec> TensorSpec::nd_shard_spec_from_legacy() {
    auto mem_layout = memory_config().memory_layout();

    if (mem_layout == TensorMemoryLayout::INTERLEAVED) {
        // Convertion from interleaved to ND sharding is not supported for now
        return std::nullopt;
    }

    if (mem_layout == TensorMemoryLayout::SINGLE_BANK) {
        // Convertion from single bank to ND sharding is not supported for now
        return std::nullopt;
    }

    const auto& shard_spec = memory_config().shard_spec().value();
    NdShardSpec nd_shard_spec{
        .physical_shard_shape = ttnn::Shape().to_rank(logical_shape_.rank()),
        .cores = shard_spec.grid,
        .shard_orientation = shard_spec.orientation,
    };

    if (mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // Can't convert logical sharding if physical shard shape is different from logical shard shape
        if (shard_spec.mode == ShardMode::LOGICAL) {
            if (shard_spec.physical_shard_shape.has_value() && *shard_spec.physical_shard_shape != shard_spec.shape) {
                return std::nullopt;
            }
        }

        if (logical_shape_.rank() >= 2) {
            nd_shard_spec.physical_shard_shape[-2] = shard_spec.shape[0];
        }
        if (logical_shape_.rank() >= 1) {
            nd_shard_spec.physical_shard_shape[-1] = shard_spec.shape[1];
        }
        return nd_shard_spec;
    }

    if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        if (logical_shape_.rank() >= 1) {
            nd_shard_spec.physical_shard_shape[-1] = logical_shape_[-1];
        }
        return nd_shard_spec;
    }

    if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        if (logical_shape_.rank() >= 2) {
            nd_shard_spec.physical_shard_shape[-2] = logical_shape_[-2];
        }
        return nd_shard_spec;
    }

    TT_FATAL(false, "Unsupported memory layout for ND sharding conversion: {}", mem_layout);
}

std::optional<ShardSpec> TensorSpec::legacy_shard_spec_from_nd() {
    const auto& nd_shard_spec = memory_config().nd_shard_spec().value();
    const auto& nd_shard_shape = nd_shard_spec.physical_shard_shape;

    // Only block sharding is supported for now
    if (memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED) {
        return std::nullopt;
    }

    // More than 2 dimensional sharding can't be converted to legacy sharding
    if (nd_shard_shape.volume() != nd_shard_shape[-1] * nd_shard_shape[-2]) {
        return std::nullopt;
    }

    ShardSpec shard_spec(
        nd_shard_spec.cores, {nd_shard_shape[-2], nd_shard_shape[-1]}, nd_shard_spec.shard_orientation);

    // Block sharding requires a contiguous grid of cores
    if (shard_spec.grid.ranges().size() != 1) {
        return std::nullopt;
    }

    // Check that the number of shards along height and width fits onto the grid
    uint32_t num_shards_along_height = div_up(physical_shape().height(), shard_spec.shape[0]);
    uint32_t num_shards_along_width = div_up(physical_shape().width(), shard_spec.shape[1]);
    if (shard_spec.orientation != ShardOrientation::ROW_MAJOR) {
        std::swap(num_shards_along_height, num_shards_along_width);
    }
    CoreCoord shard_grid = shard_spec.grid.bounding_box().grid_size();
    if (num_shards_along_height > shard_grid.x || num_shards_along_width > shard_grid.y) {
        return std::nullopt;
    }

    return shard_spec;
}

}  // namespace tt::tt_metal
