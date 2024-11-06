// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "page_config.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
size_t element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            TT_THROW("element_size_bytes() should not be used for BFLOAT8_B and BFLOAT4_B types becaues of how they are packed");

        default:
            TT_THROW("Unsupported data type!");
    }
}
} // namespace CMAKE_UNIQUE_NAMESPACE
}

PageConfig::PageConfig(const Config& config)
    : config_(config) {
}

PageConfig::PageConfig(Layout layout)
    : PageConfig(layout, std::nullopt) {
}

PageConfig::PageConfig(Layout layout, const std::optional<Tile>& tile) {
    if(layout == Layout::ROW_MAJOR) {
        config_ = RowMajorPageConfig();
    }
    else {
        config_ =  TilePageConfig(tile.value_or(Tile()));
    }
}

Alignment PageConfig::create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
    return std::visit([&](const auto& config) constexpr { return config.create_default_alignment(dtype, memory_config); }, config_);
}

void PageConfig::validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const {
    std::visit([&](const auto& config) constexpr { config.validate_alignment(alignment, dtype, memory_config); }, config_);
}

Size PageConfig::get_page_shape(const Size& physical_size, DataType dtype, const MemoryConfig& memory_config) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_shape(physical_size, dtype, memory_config); }, config_);
}

size_t PageConfig::get_page_size_bytes(const Size& page_shape, DataType dtype) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_size_bytes(page_shape, dtype); }, config_);
}

bool PageConfig::is_row_major() const {
    return std::holds_alternative<RowMajorPageConfig>(config_);
}

std::optional<Tile> PageConfig::get_tile() const
{
    if(std::holds_alternative<TilePageConfig>(config_)) {
        return std::get<TilePageConfig>(config_).get_tile();
    }

    return std::nullopt;
}


TilePageConfig::TilePageConfig(const Tile& tile)
 : tile_(tile) {
}

Alignment TilePageConfig::create_default_alignment(DataType dtype, const MemoryConfig&) const {
    return Alignment({tile_.get_height(), tile_.get_width()});
}

void TilePageConfig::validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig&) const {
    TT_FATAL(alignment.size() >= 2, "Alignment should have at least 2 dimensions for Tile layout");
    const auto widthAlignment = alignment[-1];
    TT_FATAL(widthAlignment % tile_.get_width() == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width {}.", alignment, tile_.get_width());
    auto heightAlignment = alignment[-2];
    TT_FATAL((heightAlignment % tile_.get_height()) == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of tile height {}.", alignment, tile_.get_height());
}

Size TilePageConfig::get_page_shape(const Size& physical_size, DataType dtype, const MemoryConfig& memory_config) const {
    if(memory_config.memory_layout == TensorMemoryLayout::SINGLE_BANK && physical_size.width() != 0 && physical_size.height() != 0) {
        return physical_size;
    }
    return Size(tile_.get_height(), tile_.get_width());
}

size_t TilePageConfig::get_page_size_bytes(const Size& page_shape, DataType dtype) const {
    const auto tiles_count = page_shape.height() / tile_.get_height() * page_shape.width() / tile_.get_width();
    const auto size = tiles_count * tile_.get_tile_size(datatype_to_dataformat_converter(dtype));
    return size;
}

const Tile& TilePageConfig::get_tile() const {
    return tile_;
}

Alignment RowMajorPageConfig::create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
{
    TT_FATAL(dtype != DataType::BFLOAT4_B && dtype != DataType::BFLOAT8_B, "BFLOAT4_B and BFLOAT8_B data types are not supported for ROW_MAJOR layout");

    const auto element_size = CMAKE_UNIQUE_NAMESPACE::element_size_bytes(dtype);
    auto width_alignment = sizeof(uint32_t) / element_size;

    if(memory_config.shard_spec.has_value() && memory_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        const auto shard_width = shard_shape[1];
        TT_FATAL(
            (shard_width % width_alignment) == 0,
            "Invalid sharding configuration: For Row Major layout with element size of {} bytes, the innermost dimension must align to {} bytes. "
            "Buffer data is packed as uint32_t (4 bytes), so the provided shard shape {} does not meet alignment requirements.",
            element_size, width_alignment, shard_shape
        );

        width_alignment = shard_width;
    }

    return Alignment({width_alignment});}
}

void RowMajorPageConfig::validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const {
    TT_FATAL(!alignment.empty(), "Alignment must contain at least one dimension for Row Major layout.");
    const uint32_t width_alignment = alignment[-1];
    const uint32_t element_size = CMAKE_UNIQUE_NAMESPACE::element_size_bytes(dtype);
    const uint32_t page_alignment = sizeof(uint32_t) / element_size;

    TT_FATAL((width_alignment % page_alignment) == 0,
        "Incorrect alignment configuration for Row Major layout: alignment {} requires innermost dimension alignment of {} bytes due to uint32_t (4-byte) packing, but the current alignment size is {}.",
        alignment, element_size, page_alignment);

    if(memory_config.shard_spec.has_value() && memory_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        const auto shard_width = shard_shape[1];
        TT_FATAL(
            width_alignment % shard_width == 0,
            "Alignment mismatch for sharded tensor: Expected alignment width of {} to match shard shape {} for Row Major layout.",
            width_alignment, shard_shape);
    }
}

Size RowMajorPageConfig::get_page_shape(const Size& physical_size, DataType dtype, const MemoryConfig& memory_config) const {
    if (physical_size.height() == 0 || physical_size.width() == 0) {
        return Size(1, sizeof(uint32_t) / CMAKE_UNIQUE_NAMESPACE::element_size_bytes(dtype));
    }

    if(memory_config.memory_layout == TensorMemoryLayout::SINGLE_BANK) {
        return physical_size;
    }

    if (memory_config.shard_spec.has_value() && memory_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;

        return Size(1, shard_shape[1]);
    }

    return Size(1, physical_size.width());
}

size_t RowMajorPageConfig::get_page_size_bytes(const Size& page_shape, DataType dtype) const {
    const auto size = page_shape.height() * page_shape.width() * CMAKE_UNIQUE_NAMESPACE::element_size_bytes(dtype);
    return size;
}

} // namespace tt::tt_metal
