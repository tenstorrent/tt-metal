// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_types_to_flatbuffer.hpp"

namespace ttnn {

flatbuffer::ShardOrientation to_flatbuffer(tt::tt_metal::ShardOrientation orientation) {
    switch (orientation) {
        case tt::tt_metal::ShardOrientation::ROW_MAJOR: return flatbuffer::ShardOrientation::RowMajor;
        case tt::tt_metal::ShardOrientation::COL_MAJOR: return flatbuffer::ShardOrientation::ColMajor;
    }
    TT_THROW("Unsupported ShardOrientation to flatbuffer.");
}

flatbuffer::ShardMode to_flatbuffer(tt::tt_metal::ShardMode shard_mode) {
    switch (shard_mode) {
        case tt::tt_metal::ShardMode::LOGICAL: return flatbuffer::ShardMode::Logical;
        case tt::tt_metal::ShardMode::PHYSICAL: return flatbuffer::ShardMode::Physical;
    }
    TT_THROW("Unsupported ShardMode to flatbuffer.");
}

flatbuffers::Offset<flatbuffer::ShardSpec> to_flatbuffer(
    const tt::tt_metal::ShardSpec& spec, flatbuffers::FlatBufferBuilder& builder) {
    flatbuffers::Offset<flatbuffer::ShardShape> physical_shard_shape = 0;
    if (spec.physical_shard_shape.has_value()) {
        const auto& phys_shape = *spec.physical_shard_shape;
        physical_shard_shape = flatbuffer::CreateShardShape(builder, phys_shape[0], phys_shape[1]);
    }
    return flatbuffer::CreateShardSpec(
        builder,
        to_flatbuffer(builder, spec.grid),
        spec.shape[0],
        spec.shape[1],
        to_flatbuffer(spec.orientation),
        to_flatbuffer(spec.mode),
        physical_shard_shape);
}

flatbuffers::Offset<flatbuffer::CoreCoord> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreCoord& core_coord) {
    return flatbuffer::CreateCoreCoord(builder, core_coord.x, core_coord.y);
}

flatbuffers::Offset<flatbuffer::CoreRange> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRange& core_range) {
    auto start = flatbuffer::CreateCoreCoord(builder, core_range.start_coord.x, core_range.start_coord.y);
    auto end = flatbuffer::CreateCoreCoord(builder, core_range.end_coord.x, core_range.end_coord.y);
    return flatbuffer::CreateCoreRange(builder, start, end);
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

flatbuffer::TensorMemoryLayout to_flatbuffer(tt::tt_metal::TensorMemoryLayout layout) {
    switch (layout) {
        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: return flatbuffer::TensorMemoryLayout::Interleaved;
        case tt::tt_metal::TensorMemoryLayout::SINGLE_BANK: return flatbuffer::TensorMemoryLayout::SingleBank;
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

flatbuffers::Offset<flatbuffer::MemoryConfig> to_flatbuffer(
    const tt::tt_metal::MemoryConfig& config, flatbuffers::FlatBufferBuilder& builder) {
    flatbuffers::Offset<flatbuffer::ShardSpec> shard_spec = 0;
    if (config.shard_spec.has_value()) {
        shard_spec = to_flatbuffer(*config.shard_spec, builder);
    }
    return flatbuffer::CreateMemoryConfig(
        builder, to_flatbuffer(config.memory_layout), to_flatbuffer(config.buffer_type), shard_spec);
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
            to_flatbuffer(layout.get_memory_config(), builder),
            flat_alignment);
    } else if (page_config.get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        return flatbuffer::CreateTensorLayout(
            builder,
            to_flatbuffer(layout.get_data_type()),
            flatbuffer::PageConfig::row_major,
            flatbuffer::CreateRowMajorPageConfig(builder).Union(),
            to_flatbuffer(layout.get_memory_config(), builder),
            flat_alignment);
    }
    TT_THROW("Unsupported PageConfig type to flatbuffer.");
}

flatbuffers::Offset<flatbuffer::TensorSpec> to_flatbuffer(
    const tt::tt_metal::TensorSpec& spec, flatbuffers::FlatBufferBuilder& builder) {
    const auto& shape = spec.logical_shape();
    auto flat_shape = builder.CreateVector(shape.view().data(), shape.rank());
    return flatbuffer::CreateTensorSpec(builder, flat_shape, to_flatbuffer(spec.tensor_layout(), builder));
}

}  // namespace ttnn
