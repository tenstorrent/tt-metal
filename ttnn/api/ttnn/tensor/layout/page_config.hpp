// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/alignment.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::tt_metal {

class RowMajorPageConfig {
public:
    RowMajorPageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const;
    size_t get_page_size_bytes(const Shape2D& page_size, DataType dtype) const;

    const Tile& get_tile() const;

    Alignment get_required_shard_shape_alignment() const;
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const RowMajorPageConfig&) const = default;
    bool operator!=(const RowMajorPageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    auto attribute_values() const { return std::forward_as_tuple(tile_); }

private:
    // This is currently needed for compatibility reasons.
    // Each time tile is specified, a warning will be issued. This should be removed soon.
    Tile tile_;
};

class TilePageConfig {
public:
    TilePageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const;
    size_t get_page_size_bytes(const Shape2D& page_size, DataType dtype) const;

    const Tile& get_tile() const;

    Alignment get_required_shard_shape_alignment() const;
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const TilePageConfig&) const = default;
    bool operator!=(const TilePageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    auto attribute_values() const { return std::forward_as_tuple(tile_); }

private:
    Tile tile_;
};

class PageConfig {
public:
    using Config = std::variant<RowMajorPageConfig, TilePageConfig>;

    PageConfig(const Config& config);
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    // Alignment is applied to the tensor shape, to guarantee that it is divisible by page size.
    // For tile layout, the page size is the tile size, so alignment is also equal to the tile size.
    // For row major layout, the page size is the width of the tensor, or the width of the shard (for sharded tensors).
    // So for row major tensors, alignment is either [1] for interleaved tensors or shard width for sharded tensors.
    // Note: alignment rules are different for logical sharding.
    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Shape2D get_page_shape(
        const Shape2D& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Shape2D>& physical_shard_size) const;
    size_t get_page_size_bytes(const Shape2D& page_size, DataType dtype) const;

    Tile get_tile() const;

    Layout get_layout() const;

    /// Returns the minimum required alignment for the shard shape.
    Alignment get_required_shard_shape_alignment() const;

    /// Returns the recommended alignment for the shard shape.
    /// This takes into account device memory alignment requirements trying to optimize memory usage and read/write
    /// performance. The exact device alignment requirements are dependent on device architecture and BufferType, so the
    /// maximum possible alignment is used.
    Alignment get_recommended_shard_shape_alignment(DataType dtype) const;

    bool operator==(const PageConfig&) const = default;
    bool operator!=(const PageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("config");
    auto attribute_values() const { return std::forward_as_tuple(config_); }

private:
    Config config_;
};

}  // namespace tt::tt_metal
