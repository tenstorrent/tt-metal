// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "page_config.hpp"

namespace tt::tt_metal {

namespace utils {
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
}

PageConfig::PageConfig(const Config& config)
    : m_config(config) {
}

PageConfig::PageConfig(Layout layout)
    : PageConfig(layout, std::nullopt) {
}

PageConfig::PageConfig(Layout layout, const std::optional<Tile>& tile) {
    if(layout == Layout::ROW_MAJOR) {
        m_config = RowMajorPageConfig();
    }
    else {
        m_config =  TilePageConfig(tile.value_or(Tile()));
    }
}

Alignment PageConfig::create_default_alignment(DataType dtype) const {
    return std::visit([&](const auto& config) constexpr { return config.create_default_alignment(dtype); }, m_config);
}

void PageConfig::validate_alignment(const Alignment& alignment, DataType dtype) const {
    std::visit([&](const auto& config) constexpr { config.validate_alignment(alignment, dtype); }, m_config);
}

Size PageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memory_config) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_shape(physical_size, memory_config); }, m_config);
}

size_t PageConfig::get_page_size_bytes(const Size& page_shape, DataType dtype) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_size_bytes(page_shape, dtype); }, m_config);
}

bool PageConfig::is_row_major() const {
    return std::holds_alternative<RowMajorPageConfig>(m_config);
}

std::optional<Tile> PageConfig::get_tile() const
{
    if(std::holds_alternative<TilePageConfig>(m_config)) {
        return std::get<TilePageConfig>(m_config).get_tile();
    }

    return std::nullopt;
}


TilePageConfig::TilePageConfig(const Tile& tile)
 : m_tile(tile) {
}

Alignment TilePageConfig::create_default_alignment(DataType dtype) const {
    return Alignment({m_tile.get_height(), m_tile.get_width()});
}

void TilePageConfig::validate_alignment(const Alignment& alignment, DataType dtype) const {
    TT_FATAL(alignment.size() >= 2, "Alignment should have at least 2 dimensions for Tile layout");
    const auto widthAlignment = alignment[-1];
    TT_FATAL(widthAlignment % m_tile.get_width() == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width {}.", alignment, m_tile.get_width());
    auto heightAlignment = alignment[-2];
    TT_FATAL((heightAlignment % m_tile.get_height()) == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of tile height {}.", alignment, m_tile.get_height());
}

Size TilePageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memory_config) const {
    return Size(m_tile.get_height(), m_tile.get_width());
}

size_t TilePageConfig::get_page_size_bytes(const Size& page_shape, DataType dtype) const {
    const auto tiles_count = page_shape.height() / m_tile.get_height() * page_shape.width() / m_tile.get_width();
    const auto size = tiles_count * m_tile.get_tile_size(datatype_to_dataformat_converter(dtype));
    return size;
}

const Tile& TilePageConfig::get_tile() const {
    return m_tile;
}


Alignment RowMajorPageConfig::create_default_alignment(DataType dtype) const {
{
    TT_FATAL(dtype != DataType::BFLOAT4_B && dtype != DataType::BFLOAT8_B, "BFLOAT4_B and BFLOAT8_B data types are not supported for ROW_MAJOR layout");
    return Alignment({sizeof(uint32_t) / utils::element_size_bytes(dtype)});}
}

void RowMajorPageConfig::validate_alignment(const Alignment& alignment, DataType dtype) const {
    TT_FATAL(alignment.size() > 0, "Alignment should have at least 1 dimension for Row Major layout");
    uint32_t widthAlignment = alignment[-1];
    uint32_t element_size = utils::element_size_bytes(dtype);
    uint32_t page_alignment = sizeof(uint32_t) / element_size;
    TT_FATAL((widthAlignment % page_alignment) == 0,
        "Wrong custom Tensor Layout alignment {}. For Row Major layout with element size {}bytes the innermost dimension must align to {}. This is because Buffer data is packed as uint32_t (4 bytes).",
        alignment, element_size, page_alignment);
}

Size RowMajorPageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memory_config) const {
    if (memory_config.shard_spec.has_value()) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        return Size(1, shard_shape[1]);
    }
    return Size(1, physical_size.width());
}

size_t RowMajorPageConfig::get_page_size_bytes(const Size& page_shape, DataType dtype) const {
    const auto size = page_shape.height() * page_shape.width() * utils::element_size_bytes(dtype);
    return size;
}

} // namespace tt::tt_metal
