// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/enum_types.hpp"

#include "impl/tile/tile.hpp"

#include "alignment.hpp"
#include "size.hpp"

#include <optional>

namespace tt::tt_metal {

class RowMajorPageConfig {
public:
    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Size get_page_shape(const Size& physical_size, DataType dtype, const MemoryConfig& memory_config) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;
};

class TilePageConfig {
public:
    TilePageConfig(const Tile& tile = Tile());

    Alignment create_default_alignment(DataType dtype, const MemoryConfig& memory_config) const;
    void validate_alignment(const Alignment& alignment, DataType dtype, const MemoryConfig& memory_config) const;

    Size get_page_shape(const Size& physical_size, DataType dtype, const MemoryConfig& memory_config) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    const Tile& get_tile() const;

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

    Size get_page_shape(const Size& physical_size, DataType dtype, const MemoryConfig& memory_config) const;
    size_t get_page_size_bytes(const Size& page_size, DataType dtype) const;

    std::optional<Tile> get_tile() const;

    bool is_row_major() const;

private:
    Config config_;
};

} // namespace tt::tt_metal
