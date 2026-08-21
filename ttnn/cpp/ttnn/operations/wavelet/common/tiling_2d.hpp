// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tt_stl/assert.hpp>
#include <vector>

namespace ttnn::operations::wavelet {

inline constexpr size_t kTileHeight2D = 32;
inline constexpr size_t kTileWidth2D = 32;

struct Shape2D {
    size_t height{0};
    size_t width{0};

    [[nodiscard]] constexpr bool empty() const noexcept { return height == 0 || width == 0; }
    [[nodiscard]] constexpr bool is_tile_aligned() const noexcept {
        return !empty() && height % kTileHeight2D == 0 && width % kTileWidth2D == 0;
    }

    friend constexpr bool operator==(const Shape2D&, const Shape2D&) = default;
};

struct TiledShape2D {
    Shape2D logical{};
    Shape2D storage{};

    [[nodiscard]] static TiledShape2D from_logical(Shape2D logical);

    friend constexpr bool operator==(const TiledShape2D&, const TiledShape2D&) = default;
};

struct Lwt2DTilingContract {
    TiledShape2D input{};
    TiledShape2D band{};
    bool padding_precedes_split{true};
};

[[nodiscard]] inline size_t checked_shape_area_2d(const Shape2D shape, const char* label) {
    TT_FATAL(!shape.empty(), "{} shape can't be empty, got {}x{}", label, shape.height, shape.width);
    TT_FATAL(
        shape.height <= std::numeric_limits<size_t>::max() / shape.width,
        "{} shape {}x{} overflows size_t",
        label,
        shape.height,
        shape.width);
    return shape.height * shape.width;
}

[[nodiscard]] inline size_t round_up_to_tile_2d(const size_t value, const size_t tile_extent, const char* label) {
    TT_FATAL(value > 0, "{} logical extent must be positive, got {}", label, value);
    TT_FATAL(
        value <= std::numeric_limits<size_t>::max() - (tile_extent - 1),
        "{} logical extent {} cannot be rounded to a {}-element tile",
        label,
        value,
        tile_extent);
    return ((value + tile_extent - 1) / tile_extent) * tile_extent;
}

inline TiledShape2D TiledShape2D::from_logical(const Shape2D logical) {
    static_cast<void>(checked_shape_area_2d(logical, "2D tensor logical"));
    const Shape2D storage{
        .height = round_up_to_tile_2d(logical.height, kTileHeight2D, "2D tensor height"),
        .width = round_up_to_tile_2d(logical.width, kTileWidth2D, "2D tensor width"),
    };
    static_cast<void>(checked_shape_area_2d(storage, "2D tensor storage"));
    return TiledShape2D{.logical = logical, .storage = storage};
}

inline void validate_tiled_shape_2d(const TiledShape2D& shape, const char* label) {
    static_cast<void>(checked_shape_area_2d(shape.logical, label));
    static_cast<void>(checked_shape_area_2d(shape.storage, label));
    TT_FATAL(
        shape.storage.is_tile_aligned(),
        "{} storage shape {}x{} violates the 32x32 tiling contract",
        label,
        shape.storage.height,
        shape.storage.width);
    TT_FATAL(
        shape.logical.height <= shape.storage.height && shape.logical.width <= shape.storage.width,
        "{} logical shape {}x{} exceeds storage shape {}x{}",
        label,
        shape.logical.height,
        shape.logical.width,
        shape.storage.height,
        shape.storage.width);
    TT_FATAL(
        shape == TiledShape2D::from_logical(shape.logical),
        "{} storage shape {}x{} is not the minimal 32x32 expansion of logical shape {}x{}",
        label,
        shape.storage.height,
        shape.storage.width,
        shape.logical.height,
        shape.logical.width);
}

inline void validate_lwt_2d_tiling_contract(const Lwt2DTilingContract& contract) {
    TT_FATAL(contract.padding_precedes_split, "2D input padding must be applied before split2d");
    validate_tiled_shape_2d(contract.input, "2D LWT input");
    validate_tiled_shape_2d(contract.band, "2D LWT output band");
}

[[nodiscard]] inline std::vector<float> zero_pad_row_major_to_tiles_2d(
    const std::span<const float> input, const Shape2D logical_shape) {
    const TiledShape2D tiled_shape = TiledShape2D::from_logical(logical_shape);
    const size_t logical_elements = checked_shape_area_2d(logical_shape, "2D input");
    TT_FATAL(
        input.size() == logical_elements,
        "2D input has {} elements but logical shape {}x{} requires {}",
        input.size(),
        logical_shape.height,
        logical_shape.width,
        logical_elements);
    std::vector<float> padded(checked_shape_area_2d(tiled_shape.storage, "2D padded input"), 0.0F);
    for (size_t row = 0; row < logical_shape.height; ++row) {
        std::copy_n(
            input.begin() + static_cast<std::ptrdiff_t>(row * logical_shape.width),
            logical_shape.width,
            padded.begin() + static_cast<std::ptrdiff_t>(row * tiled_shape.storage.width));
    }
    return padded;
}

[[nodiscard]] inline bool has_zero_tile_padding_2d(const std::span<const float> padded, const TiledShape2D& shape) {
    validate_tiled_shape_2d(shape, "2D padded input");
    if (padded.size() != checked_shape_area_2d(shape.storage, "2D padded input")) {
        return false;
    }
    for (size_t row = 0; row < shape.storage.height; ++row) {
        for (size_t column = 0; column < shape.storage.width; ++column) {
            if ((row >= shape.logical.height || column >= shape.logical.width) &&
                padded[row * shape.storage.width + column] != 0.0F) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace ttnn::operations::wavelet
