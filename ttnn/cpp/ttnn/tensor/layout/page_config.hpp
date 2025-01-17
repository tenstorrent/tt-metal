// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/enum_types.hpp"

#include <tt-metalium/tile.hpp>

#include "alignment.hpp"
#include "size.hpp"

#include <optional>

namespace tt::tt_metal {

class RowMajorPageConfig {
public:
    RowMajorPageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Size get_page_shape(
        const Size& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Size>& physical_shard_size) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    const Tile& get_tile() const;

    bool operator==(const RowMajorPageConfig&) const = default;
    bool operator!=(const RowMajorPageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    const auto attribute_values() const { return std::forward_as_tuple(tile_); }

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

    Size get_page_shape(
        const Size& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Size>& physical_shard_size) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    const Tile& get_tile() const;

    bool operator==(const TilePageConfig&) const = default;
    bool operator!=(const TilePageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("tile");
    const auto attribute_values() const { return std::forward_as_tuple(tile_); }

private:
    Tile tile_;
};

class PageConfig {
public:
    using Config = std::variant<RowMajorPageConfig, TilePageConfig>;

    PageConfig(const Config& config);
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Size get_page_shape(
        const Size& physical_size,
        DataType dtype,
        const MemoryConfig& memory_config,
        const std::optional<Size>& physical_shard_size) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    Tile get_tile() const;

    Layout get_layout() const;

    bool operator==(const PageConfig&) const = default;
    bool operator!=(const PageConfig&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("config");
    const auto attribute_values() const { return std::forward_as_tuple(config_); }

private:
    Config config_;
};

}  // namespace tt::tt_metal
