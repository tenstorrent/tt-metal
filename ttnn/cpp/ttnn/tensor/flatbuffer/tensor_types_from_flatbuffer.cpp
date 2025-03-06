// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_types_from_flatbuffer.hpp"

namespace ttnn {

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
        case flatbuffer::TensorMemoryLayout::SingleBank: return tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
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

tt::tt_metal::MemoryConfig from_flatbuffer(const flatbuffer::MemoryConfig* config) {
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
    if (config->shard_spec()) {
        shard_spec = from_flatbuffer(config->shard_spec());
    }
    return tt::tt_metal::MemoryConfig{
        from_flatbuffer(config->memory_layout()),
        from_flatbuffer(config->buffer_type()),
        shard_spec,
    };
}

tt::tt_metal::ShardOrientation from_flatbuffer(flatbuffer::ShardOrientation orientation) {
    switch (orientation) {
        case flatbuffer::ShardOrientation::RowMajor: return tt::tt_metal::ShardOrientation::ROW_MAJOR;
        case flatbuffer::ShardOrientation::ColMajor: return tt::tt_metal::ShardOrientation::COL_MAJOR;
    }
    TT_THROW("Unsupported ShardOrientation from flatbuffer.");
}

tt::tt_metal::ShardMode from_flatbuffer(flatbuffer::ShardMode mode) {
    switch (mode) {
        case flatbuffer::ShardMode::Physical: return tt::tt_metal::ShardMode::PHYSICAL;
        case flatbuffer::ShardMode::Logical: return tt::tt_metal::ShardMode::LOGICAL;
    }
    TT_THROW("Unsupported ShardMode from flatbuffer.");
}

tt::tt_metal::ShardSpec from_flatbuffer(const flatbuffer::ShardSpec* spec) {
    CoreRangeSet grid = from_flatbuffer(spec->grid());
    std::array<uint32_t, 2> shape = {spec->shape_h(), spec->shape_w()};
    tt::tt_metal::ShardOrientation orientation = from_flatbuffer(spec->orientation());
    tt::tt_metal::ShardMode mode = from_flatbuffer(spec->shard_mode());
    if (const auto* fb_shard_shape = spec->physical_shard_shape()) {
        std::array<uint32_t, 2> physical_shard_shape = {fb_shard_shape->height(), fb_shard_shape->width()};
        return tt::tt_metal::ShardSpec(grid, shape, physical_shard_shape, orientation);
    }
    return tt::tt_metal::ShardSpec(grid, shape, orientation, mode);
}

CoreCoord from_flatbuffer(const flatbuffer::CoreCoord* core_coord) {
    return CoreCoord{core_coord->x(), core_coord->y()};
}

CoreRange from_flatbuffer(const flatbuffer::CoreRange* core_range) {
    return CoreRange{
        {core_range->start()->x(), core_range->start()->y()}, {core_range->end()->x(), core_range->end()->y()}};
}

CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* core_range_set) {
    std::vector<CoreRange> ranges;
    for (const auto* range : *core_range_set->ranges()) {
        ranges.emplace_back(
            CoreCoord{range->start()->x(), range->start()->y()}, CoreCoord{range->end()->x(), range->end()->y()});
    }
    return CoreRangeSet{ranges};
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
        from_flatbuffer(layout->memory_config()),
        tt::tt_metal::Alignment(SmallVector<uint32_t>(layout->alignment()->cbegin(), layout->alignment()->cend())));
}

tt::tt_metal::TensorSpec from_flatbuffer(const flatbuffer::TensorSpec* spec) {
    return tt::tt_metal::TensorSpec(
        Shape(SmallVector<uint32_t>(spec->shape()->cbegin(), spec->shape()->cend())),
        from_flatbuffer(spec->tensor_layout()));
}

}  // namespace ttnn
