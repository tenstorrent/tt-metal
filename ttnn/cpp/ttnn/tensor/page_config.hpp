// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "impl/tile/tile.hpp"
#include "size.hpp"

namespace tt::tt_metal {

class RowMajorPageConfig {
public:
    Alignment create_default_alignment(DataType dtype) const;
    void validate_alignment(const Alignment& alignment, DataType dtype) const;

    Size get_page_shape(const Size& physical_size, const MemoryConfig& memory_config) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;
};

class TilePageConfig {
public:
    TilePageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype) const;
    void validate_alignment(const Alignment& alignment, DataType dtype) const;

    Size get_page_shape(const Size& physical_size, const MemoryConfig& memory_config) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    const Tile& get_tile() const;

private:
    Tile m_tile;
};

class PageConfig {
public:
    using Config = std::variant<RowMajorPageConfig, TilePageConfig>;

    PageConfig(const Config& config);
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    Alignment create_default_alignment(DataType dtype) const;
    void validate_alignment(const Alignment& alignment, DataType dtype) const;

    Size get_page_shape(const Size& physical_size, const MemoryConfig& memory_config) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    std::optional<Tile> get_tile() const;

    bool is_row_major() const;

private:
    Config m_config;
};

} // namespace tt::tt_metal
