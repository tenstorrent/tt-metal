// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/layout/page_config.hpp"

#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
size_t rm_element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            // To store block floats in RowMajor layout, we use a fallback and store full floats instead
            return sizeof(float);

        default: TT_THROW("Unsupported data type!");
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

PageConfig::PageConfig(const Config& config) : config_(config) {}

PageConfig::PageConfig(Layout layout) : PageConfig(layout, std::nullopt) {}

PageConfig::PageConfig(Layout layout, const std::optional<Tile>& tile) {
    if (layout == Layout::ROW_MAJOR) {
        // TODO: add TT_FATAL(!tile.has_value(), "Specifying tile shape for a row major layout is not supported")
        config_ = RowMajorPageConfig(tile.value_or(Tile()));
    } else {
        config_ = TilePageConfig(tile.value_or(Tile()));
    }
}

Alignment PageConfig::create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
    return std::visit(
        [&](const auto& config) constexpr { return config.create_default_alignment(dtype, memory_config); }, config_);
}

void PageConfig::validate_alignment(
    const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const {
    std::visit(
        [&](const auto& config) constexpr { config.validate_alignment(alignment, dtype, memory_config); }, config_);
}

Shape2D PageConfig::get_page_shape(
    const Shape2D& physical_size,
    DataType dtype,
    const MemoryConfig& memory_config,
    const std::optional<Shape2D>& physical_shard_size) const {
    return std::visit(
        [&](const auto& config) constexpr {
            return config.get_page_shape(physical_size, dtype, memory_config, physical_shard_size);
        },
        config_);
}

size_t PageConfig::get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const {
    return std::visit(
        [&](const auto& config) constexpr { return config.get_page_size_bytes(page_shape, dtype); }, config_);
}

Layout PageConfig::get_layout() const {
    if (std::holds_alternative<RowMajorPageConfig>(config_)) {
        return Layout::ROW_MAJOR;
    }
    return Layout::TILE;
}

Tile PageConfig::get_tile() const {
    return std::visit([&](const auto& config) { return config.get_tile(); }, config_);
}

TilePageConfig::TilePageConfig(const Tile& tile) : tile_(tile) {}

Alignment TilePageConfig::create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
    if (memory_config.shard_spec().has_value()) {
        const auto& shard_spec = memory_config.shard_spec().value();
        if (shard_spec.physical_shard_shape.has_value()) {
            return Alignment(shard_spec.physical_shard_shape.value());
        }
    }
    return Alignment({tile_.get_height(), tile_.get_width()});
}

void TilePageConfig::validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig&) const {
    TT_FATAL(alignment.size() >= 2, "Alignment should have at least 2 dimensions for Tile layout");
    const auto widthAlignment = alignment[-1];
    TT_FATAL(
        widthAlignment % tile_.get_width() == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width "
        "{}.",
        alignment,
        tile_.get_width());
    auto heightAlignment = alignment[-2];
    TT_FATAL(
        (heightAlignment % tile_.get_height()) == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of "
        "tile height {}.",
        alignment,
        tile_.get_height());
}

Shape2D TilePageConfig::get_page_shape(
    const Shape2D& physical_size,
    DataType dtype,
    const MemoryConfig& memory_config,
    const std::optional<Shape2D>&) const {
    if (memory_config.memory_layout() == TensorMemoryLayout::SINGLE_BANK && physical_size.width() != 0 &&
        physical_size.height() != 0) {
        return physical_size;
    }
    return Shape2D(tile_.get_height(), tile_.get_width());
}

size_t TilePageConfig::get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const {
    const auto tiles_count = page_shape.height() / tile_.get_height() * page_shape.width() / tile_.get_width();
    const auto size = tiles_count * tile_.get_tile_size(datatype_to_dataformat_converter(dtype));
    return size;
}

const Tile& TilePageConfig::get_tile() const { return tile_; }

RowMajorPageConfig::RowMajorPageConfig(const Tile& tile) : tile_(tile) {}

Alignment RowMajorPageConfig::create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const {
    if (memory_config.shard_spec().has_value()) {
        const auto& shard_spec = memory_config.shard_spec().value();
        if (shard_spec.mode == ShardMode::LOGICAL) {
            return shard_spec.physical_shard_shape.has_value() ? Alignment(shard_spec.physical_shard_shape.value())
                                                               : Alignment({shard_spec.shape[1]});
        }
        // TODO: Investigate why we need guard against HEIGHT_SHARDED and merge logic with LOGICAL sharding
        if (shard_spec.mode == ShardMode::PHYSICAL &&
            memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
            return Alignment({shard_spec.shape[1]});
        }
    } else if (memory_config.nd_shard_spec().has_value()) {
        const auto& nd_shard_spec = *memory_config.nd_shard_spec();
        return Alignment({nd_shard_spec.shard_shape[-1]});
    }
    return Alignment({1});
}

void RowMajorPageConfig::validate_alignment(
    const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const {
    TT_FATAL(!alignment.empty(), "Alignment must contain at least one dimension for Row Major layout.");
    const uint32_t width_alignment = alignment[-1];

    // TODO: Do we need to validate sharded width here if wee are guaranteed that physical_shard_width is set as
    // width_alignment
    if (memory_config.shard_spec().has_value() && memory_config.shard_spec().value().mode == ShardMode::PHYSICAL &&
        memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& physical_shard_shape = memory_config.shard_spec().value().shape;
        const auto physical_shard_width = physical_shard_shape[1];
        TT_FATAL(
            physical_shard_width % width_alignment == 0,
            "Alignment mismatch for sharded tensor: Expected physical shard shape {} to be aligned to {} along the "
            "width for Row Major layout.",
            physical_shard_width,
            width_alignment);
    }
}

Shape2D RowMajorPageConfig::get_page_shape(
    const Shape2D& physical_size,
    DataType dtype,
    const MemoryConfig& memory_config,
    const std::optional<Shape2D>& physical_shard_size) const {
    if (physical_size.height() == 0 || physical_size.width() == 0) {
        return Shape2D(1, sizeof(uint32_t) / CMAKE_UNIQUE_NAMESPACE::rm_element_size_bytes(dtype));
    }

    if (memory_config.memory_layout() == TensorMemoryLayout::SINGLE_BANK) {
        return physical_size;
    }

    if (memory_config.shard_spec().has_value() && memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(
            physical_shard_size.has_value(),
            "For width or block sharded tensors, Row Major page width comes from physical shard size so it must be "
            "provided!");

        return Shape2D(1, physical_shard_size.value().width());
    }

    if (memory_config.is_sharded() && memory_config.nd_shard_spec().has_value()) {
        const auto& nd_shard_spec = *memory_config.nd_shard_spec();
        return Shape2D(1, nd_shard_spec.shard_shape[-1]);
    }

    return Shape2D(1, physical_size.width());
}

size_t RowMajorPageConfig::get_page_size_bytes(const Shape2D& page_shape, DataType dtype) const {
    const auto size = page_shape.height() * page_shape.width() * CMAKE_UNIQUE_NAMESPACE::rm_element_size_bytes(dtype);
    return size;
}

const Tile& RowMajorPageConfig::get_tile() const { return tile_; }

}  // namespace tt::tt_metal
