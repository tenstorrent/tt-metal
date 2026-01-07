// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"

namespace ttnn {
namespace {

CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* core_range_set) {
    std::vector<CoreRange> ranges;
    for (const auto* range : *core_range_set->ranges()) {
        ranges.emplace_back(
            CoreCoord{range->start()->x(), range->start()->y()}, CoreCoord{range->end()->x(), range->end()->y()});
    }
    return CoreRangeSet{ranges};
}

flatbuffers::Offset<flatbuffer::CoreRangeSet> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRangeSet& core_range_set) {
    std::vector<flatbuffers::Offset<flatbuffer::CoreRange>> range_offsets;
    for (const auto& range : core_range_set.ranges()) {
        auto start = flatbuffer::CreateCoreCoord(builder, range.start_coord.x, range.start_coord.y);
        auto end = flatbuffer::CreateCoreCoord(builder, range.end_coord.x, range.end_coord.y);
        range_offsets.push_back(flatbuffer::CreateCoreRange(builder, start, end));
    }
    auto ranges_vector = builder.CreateVector(range_offsets);
    return flatbuffer::CreateCoreRangeSet(builder, ranges_vector);
}

tt::tt_metal::BufferType from_flatbuffer(flatbuffer::BufferType type) {
    switch (type) {
        case flatbuffer::BufferType::DRAM: return tt::tt_metal::BufferType::DRAM;
        case flatbuffer::BufferType::L1: return tt::tt_metal::BufferType::L1;
        case flatbuffer::BufferType::SystemMemory: return tt::tt_metal::BufferType::SYSTEM_MEMORY;
        case flatbuffer::BufferType::L1Small: return tt::tt_metal::BufferType::L1_SMALL;
        case flatbuffer::BufferType::Trace: return tt::tt_metal::BufferType::TRACE;
    }
    TT_THROW("Unsupported BufferType from flatbuffer.");
}

tt::tt_metal::TensorMemoryLayout from_flatbuffer(flatbuffer::TensorMemoryLayout layout) {
    switch (layout) {
        case flatbuffer::TensorMemoryLayout::Interleaved: return tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
        case flatbuffer::TensorMemoryLayout::HeightSharded: return tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
        case flatbuffer::TensorMemoryLayout::WidthSharded: return tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        case flatbuffer::TensorMemoryLayout::BlockSharded: return tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
    }
    TT_THROW("Unsupported TensorMemoryLayout from flatbuffer.");
}

tt::tt_metal::DataType from_flatbuffer(flatbuffer::DataType type) {
    switch (type) {
        case flatbuffer::DataType::BFloat16: return tt::tt_metal::DataType::BFLOAT16;
        case flatbuffer::DataType::Float32: return tt::tt_metal::DataType::FLOAT32;
        case flatbuffer::DataType::UInt32: return tt::tt_metal::DataType::UINT32;
        case flatbuffer::DataType::BFloat8B: return tt::tt_metal::DataType::BFLOAT8_B;
        case flatbuffer::DataType::BFloat4B: return tt::tt_metal::DataType::BFLOAT4_B;
        case flatbuffer::DataType::UInt8: return tt::tt_metal::DataType::UINT8;
        case flatbuffer::DataType::UInt16: return tt::tt_metal::DataType::UINT16;
        case flatbuffer::DataType::Int32: return tt::tt_metal::DataType::INT32;
        case flatbuffer::DataType::Invalid: return tt::tt_metal::DataType::INVALID;
    }
    TT_THROW("Unsupported DataType from flatbuffer.");
}

flatbuffer::TensorMemoryLayout to_flatbuffer(tt::tt_metal::TensorMemoryLayout layout) {
    switch (layout) {
        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: return flatbuffer::TensorMemoryLayout::Interleaved;
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: return flatbuffer::TensorMemoryLayout::HeightSharded;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: return flatbuffer::TensorMemoryLayout::WidthSharded;
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: return flatbuffer::TensorMemoryLayout::BlockSharded;
    }
    TT_THROW("Unsupported TensorMemoryLayout to flatbuffer.");
}

flatbuffer::BufferType to_flatbuffer(tt::tt_metal::BufferType type) {
    switch (type) {
        case tt::tt_metal::BufferType::DRAM: return flatbuffer::BufferType::DRAM;
        case tt::tt_metal::BufferType::L1: return flatbuffer::BufferType::L1;
        case tt::tt_metal::BufferType::SYSTEM_MEMORY: return flatbuffer::BufferType::SystemMemory;
        case tt::tt_metal::BufferType::L1_SMALL: return flatbuffer::BufferType::L1Small;
        case tt::tt_metal::BufferType::TRACE: return flatbuffer::BufferType::Trace;
    }
    TT_THROW("Unsupported BufferType to flatbuffer.");
}

flatbuffer::DataType to_flatbuffer(tt::tt_metal::DataType type) {
    switch (type) {
        case tt::tt_metal::DataType::BFLOAT16: return flatbuffer::DataType::BFloat16;
        case tt::tt_metal::DataType::FLOAT32: return flatbuffer::DataType::Float32;
        case tt::tt_metal::DataType::UINT32: return flatbuffer::DataType::UInt32;
        case tt::tt_metal::DataType::BFLOAT8_B: return flatbuffer::DataType::BFloat8B;
        case tt::tt_metal::DataType::BFLOAT4_B: return flatbuffer::DataType::BFloat4B;
        case tt::tt_metal::DataType::UINT8: return flatbuffer::DataType::UInt8;
        case tt::tt_metal::DataType::UINT16: return flatbuffer::DataType::UInt16;
        case tt::tt_metal::DataType::INT32: return flatbuffer::DataType::Int32;
        case tt::tt_metal::DataType::INVALID: return flatbuffer::DataType::Invalid;
    }
    TT_THROW("Unsupported DataType to flatbuffer.");
}

tt::tt_metal::ShardOrientation from_flatbuffer(flatbuffer::ShardOrientation orientation) {
    switch (orientation) {
        case flatbuffer::ShardOrientation::RowMajor: return tt::tt_metal::ShardOrientation::ROW_MAJOR;
        case flatbuffer::ShardOrientation::ColMajor: return tt::tt_metal::ShardOrientation::COL_MAJOR;
    }
    TT_THROW("Unsupported ShardOrientation from flatbuffer.");
}

flatbuffer::ShardOrientation to_flatbuffer(tt::tt_metal::ShardOrientation orientation) {
    switch (orientation) {
        case tt::tt_metal::ShardOrientation::ROW_MAJOR: return flatbuffer::ShardOrientation::RowMajor;
        case tt::tt_metal::ShardOrientation::COL_MAJOR: return flatbuffer::ShardOrientation::ColMajor;
    }
    TT_THROW("Unsupported ShardOrientation to flatbuffer.");
}

flatbuffer::ShardDistributionStrategy to_flatbuffer(tt::tt_metal::ShardDistributionStrategy strategy) {
    switch (strategy) {
        case tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D:
            return flatbuffer::ShardDistributionStrategy::ROUND_ROBIN_1D;
        case tt::tt_metal::ShardDistributionStrategy::GRID_2D: return flatbuffer::ShardDistributionStrategy::GRID_2D;
    }
    TT_THROW("Unsupported ShardDistributionStrategy to flatbuffer.");
}

tt::tt_metal::ShardDistributionStrategy from_flatbuffer(flatbuffer::ShardDistributionStrategy strategy) {
    switch (strategy) {
        case flatbuffer::ShardDistributionStrategy::ROUND_ROBIN_1D:
            return tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D;
        case flatbuffer::ShardDistributionStrategy::GRID_2D: return tt::tt_metal::ShardDistributionStrategy::GRID_2D;
    }
    TT_THROW("Unsupported ShardDistributionStrategy from flatbuffer.");
}

tt::tt_metal::ShardSpec from_flatbuffer(const flatbuffer::ShardSpec* spec) {
    CoreRangeSet grid = from_flatbuffer(spec->grid());
    std::array<uint32_t, 2> shape = {spec->shape_h(), spec->shape_w()};
    tt::tt_metal::ShardOrientation orientation = from_flatbuffer(spec->orientation());
    return tt::tt_metal::ShardSpec(grid, shape, orientation);
}

flatbuffers::Offset<flatbuffer::ShardSpec> to_flatbuffer(
    const tt::tt_metal::ShardSpec& spec, flatbuffers::FlatBufferBuilder& builder) {
    flatbuffers::Offset<flatbuffer::ShardShape> physical_shard_shape = 0;
    return flatbuffer::CreateShardSpec(
        builder,
        to_flatbuffer(builder, spec.grid),
        spec.shape[0],
        spec.shape[1],
        to_flatbuffer(spec.orientation),
        flatbuffer::ShardModeDeprecated::Physical,
        physical_shard_shape);
}

flatbuffers::Offset<flatbuffer::NdShardSpec> to_flatbuffer(
    const tt::tt_metal::NdShardSpec& spec, flatbuffers::FlatBufferBuilder& builder) {
    auto flat_shape = builder.CreateVector(spec.shard_shape.view().data(), spec.shard_shape.rank());
    return flatbuffer::CreateNdShardSpec(
        builder,
        flat_shape,
        to_flatbuffer(builder, spec.grid),
        to_flatbuffer(spec.orientation),
        to_flatbuffer(spec.shard_distribution_strategy));
}

tt::tt_metal::NdShardSpec from_flatbuffer(const flatbuffer::NdShardSpec* spec) {
    return tt::tt_metal::NdShardSpec(
        Shape(SmallVector<uint32_t>(spec->shard_shape()->cbegin(), spec->shard_shape()->cend())),
        from_flatbuffer(spec->grid()),
        from_flatbuffer(spec->orientation()),
        from_flatbuffer(spec->shard_distribution_strategy()));
}

tt::tt_metal::TensorLayout from_flatbuffer(const flatbuffer::TensorLayout* layout) {
    tt::tt_metal::PageConfig page_config = [&] {
        switch (layout->page_config_type()) {
            case flatbuffer::PageConfig::row_major: return tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
            case flatbuffer::PageConfig::tile: {
                const auto* tile_page_config = layout->page_config_as_tile();
                const auto* flat_tile = tile_page_config->tile();
                tt::tt_metal::Tile tile(
                    std::array{flat_tile->tile_shape_h(), flat_tile->tile_shape_w()}, flat_tile->transpose_tile());
                return tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, tile);
            }
            default: TT_THROW("Unsupported PageConfig type from flatbuffer.");
        }
    }();

    return tt::tt_metal::TensorLayout::restore_from_serialized(
        from_flatbuffer(layout->data_type()),
        page_config,
        ttnn::from_flatbuffer(layout->memory_config()),
        tt::tt_metal::Alignment(SmallVector<uint32_t>(layout->alignment()->cbegin(), layout->alignment()->cend())));
}

flatbuffers::Offset<flatbuffer::TensorLayout> to_flatbuffer(
    const tt::tt_metal::TensorLayout& layout, flatbuffers::FlatBufferBuilder& builder) {
    const auto& alignment = layout.get_alignment();
    auto flat_alignment = builder.CreateVector(alignment.view().data(), alignment.size());
    auto page_config = layout.get_page_config();
    if (page_config.get_layout() == tt::tt_metal::Layout::TILE) {
        auto tile = page_config.get_tile();
        auto flat_tile =
            flatbuffer::CreateTile(builder, tile.get_height(), tile.get_width(), tile.get_transpose_of_faces());
        return flatbuffer::CreateTensorLayout(
            builder,
            to_flatbuffer(layout.get_data_type()),
            flatbuffer::PageConfig::tile,
            flatbuffer::CreateTilePageConfig(builder, flat_tile).Union(),
            ttnn::to_flatbuffer(layout.get_memory_config(), builder),
            flat_alignment);
    }
    if (page_config.get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        return flatbuffer::CreateTensorLayout(
            builder,
            to_flatbuffer(layout.get_data_type()),
            flatbuffer::PageConfig::row_major,
            flatbuffer::CreateRowMajorPageConfig(builder).Union(),
            ttnn::to_flatbuffer(layout.get_memory_config(), builder),
            flat_alignment);
    }
    TT_THROW("Unsupported PageConfig type to flatbuffer.");
}

}  // namespace

flatbuffers::Offset<flatbuffer::MemoryConfig> to_flatbuffer(
    const tt::tt_metal::MemoryConfig& config, flatbuffers::FlatBufferBuilder& builder) {
    flatbuffers::Offset<flatbuffer::ShardSpec> shard_spec = 0;
    flatbuffers::Offset<flatbuffer::NdShardSpec> nd_shard_spec = 0;
    if (config.shard_spec().has_value()) {
        shard_spec = to_flatbuffer(*config.shard_spec(), builder);
    }
    if (config.nd_shard_spec().has_value()) {
        nd_shard_spec = to_flatbuffer(*config.nd_shard_spec(), builder);
    }
    return flatbuffer::CreateMemoryConfig(
        builder,
        to_flatbuffer(config.memory_layout()),
        to_flatbuffer(config.buffer_type()),
        shard_spec,
        nd_shard_spec,
        config.created_with_nd_shard_spec());
}

tt::tt_metal::MemoryConfig from_flatbuffer(const flatbuffer::MemoryConfig* config) {
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
    std::optional<tt::tt_metal::NdShardSpec> nd_shard_spec;
    if (config->shard_spec()) {
        shard_spec = from_flatbuffer(config->shard_spec());
    }
    if (config->nd_shard_spec()) {
        nd_shard_spec = from_flatbuffer(config->nd_shard_spec());
    }
    return tt::tt_metal::MemoryConfig::create_with_prepopulated_shard_specs(
        from_flatbuffer(config->memory_layout()),
        from_flatbuffer(config->buffer_type()),
        shard_spec,
        nd_shard_spec,
        config->created_with_nd_shard_spec());
}

flatbuffers::Offset<flatbuffer::TensorSpec> to_flatbuffer(
    const tt::tt_metal::TensorSpec& spec, flatbuffers::FlatBufferBuilder& builder) {
    const auto& shape = spec.logical_shape();
    auto flat_shape = builder.CreateVector(shape.view().data(), shape.rank());
    return flatbuffer::CreateTensorSpec(builder, flat_shape, to_flatbuffer(spec.tensor_layout(), builder));
}

tt::tt_metal::TensorSpec from_flatbuffer(const flatbuffer::TensorSpec* spec) {
    return tt::tt_metal::TensorSpec(
        Shape(SmallVector<uint32_t>(spec->shape()->cbegin(), spec->shape()->cend())),
        from_flatbuffer(spec->tensor_layout()));
}

}  // namespace ttnn
