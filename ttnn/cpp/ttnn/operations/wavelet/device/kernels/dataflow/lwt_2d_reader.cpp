// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "../primitives/config_page.hpp"
#include "../primitives/noc_local.hpp"
#include "../primitives/tile_2d_layout.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/wavelet/common/signal_extension.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_2d_config.hpp"
#include "ttnn/operations/wavelet/planner/step.hpp"

namespace {

using ttnn::operations::wavelet::kernels::primitives::ConfigWords;
using ttnn::operations::wavelet::kernels::primitives::kFaceSide;
using ttnn::operations::wavelet::kernels::primitives::kTileBytes;
using ttnn::operations::wavelet::kernels::primitives::kTileElements;
using ttnn::operations::wavelet::kernels::primitives::kTileSide;
using ttnn::operations::wavelet::kernels::primitives::load_config_page;
using ttnn::operations::wavelet::kernels::primitives::tile_element_offset;
using ttnn::operations::wavelet::kernels::primitives::tile_face_column_offset;
using ttnn::operations::wavelet::kernels::primitives::tile_face_row_offset;
using ttnn::operations::wavelet::kernels::primitives::tiled_element_offset;

struct Rect {
    uint32_t y_begin{0};
    uint32_t y_length{0};
    uint32_t x_begin{0};
    uint32_t x_length{0};

    Rect() = default;

    ALWI Rect(const ConfigWords words, const uint32_t offset) :
        y_begin(words[offset + ttnn::operations::wavelet::device_protocol::kLwt2DRectYBegin]),
        y_length(words[offset + ttnn::operations::wavelet::device_protocol::kLwt2DRectYLength]),
        x_begin(words[offset + ttnn::operations::wavelet::device_protocol::kLwt2DRectXBegin]),
        x_length(words[offset + ttnn::operations::wavelet::device_protocol::kLwt2DRectXLength]) {}
};

#if defined(LWT_2D_COMPACT_BOUNDARY_CODE) || defined(ILWT_2D_COMPACT_BOUNDARY_CODE)
#define LWT_2D_BOUNDARY_FUNCTION __attribute__((noinline))
#else
#define LWT_2D_BOUNDARY_FUNCTION ALWI
#endif

[[nodiscard]] ALWI uint32_t aligned_begin(const uint32_t value) { return (value / kTileSide) * kTileSide; }

[[nodiscard]] ALWI uint32_t aligned_end(const uint32_t begin, const uint32_t length) {
    return ((begin + length + kTileSide - 1) / kTileSide) * kTileSide;
}

[[nodiscard]] ALWI bool contains(const Rect& rectangle, const int32_t y, const int32_t x) {
    return y >= static_cast<int32_t>(rectangle.y_begin) && x >= static_cast<int32_t>(rectangle.x_begin) &&
           y < static_cast<int32_t>(rectangle.y_begin + rectangle.y_length) &&
           x < static_cast<int32_t>(rectangle.x_begin + rectangle.x_length);
}

template <typename Accessor>
ALWI void preload_config_pages(
    const Accessor& accessor,
    const uint32_t address,
    const uint32_t page_bytes,
    const uint32_t page_begin,
    const uint32_t page_count,
    const uint32_t destination_addr) {
    const auto pages = TensorAccessor(accessor, address, page_bytes);
    Noc noc;
    for (uint32_t page = 0; page < page_count; ++page) {
        noc.async_read(
            pages,
            CoreLocalMem<uint32_t>(destination_addr + page * page_bytes),
            page_bytes,
            {.page_id = page_begin + page},
            {});
    }
    noc.async_read_barrier();
}

template <ttnn::operations::wavelet::BoundaryMode Mode>
struct SplitSourceTiles {
    static constexpr uint32_t kAxisCapacity =
        Mode == ttnn::operations::wavelet::BoundaryMode::kSymmetric
            ? ttnn::operations::wavelet::device_protocol::kLwt2DSymmetricSplitScratchTileRows
            : ttnn::operations::wavelet::device_protocol::kLwt2DSplitScratchTileRows;
    uint32_t rows[kAxisCapacity];
    uint32_t columns[kAxisCapacity];
    uint32_t row_count;
    uint32_t column_count;
};

struct SourceAxisTileCollector {
    uint32_t* tiles;
    uint32_t& count;
    uint32_t capacity;

    LWT_2D_BOUNDARY_FUNCTION void operator()(const uint32_t source_index) const {
        const uint32_t source_tile = source_index / kTileSide;
        for (uint32_t index = 0; index < count; ++index) {
            if (tiles[index] == source_tile) {
                return;
            }
        }
        if (count < capacity) {
            tiles[count++] = source_tile;
        }
    }
};

[[nodiscard]] ALWI bool intersects_tile(const Rect& rectangle, const uint32_t tile_y, const uint32_t tile_x) {
    return rectangle.y_begin < tile_y + kTileSide && rectangle.y_begin + rectangle.y_length > tile_y &&
           rectangle.x_begin < tile_x + kTileSide && rectangle.x_begin + rectangle.x_length > tile_x;
}

[[nodiscard]] ALWI bool covers_tile(const Rect& rectangle, const uint32_t tile_y, const uint32_t tile_x) {
    return rectangle.y_begin <= tile_y && rectangle.y_begin + rectangle.y_length >= tile_y + kTileSide &&
           rectangle.x_begin <= tile_x && rectangle.x_begin + rectangle.x_length >= tile_x + kTileSide;
}

template <uint32_t Capacity>
[[nodiscard]] ALWI uint32_t find_index(const uint32_t (&values)[Capacity], const uint32_t count, const uint32_t value) {
    for (uint32_t index = 0; index < count; ++index) {
        if (values[index] == value) {
            return index;
        }
    }
    return 0;
}

template <ttnn::operations::wavelet::BoundaryMode Mode>
LWT_2D_BOUNDARY_FUNCTION void collect_boundary_source_axis_tiles(
    uint32_t* tiles, uint32_t& count, const uint32_t capacity, const int32_t raw_begin, const uint32_t logical_length) {
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kSymmetric) {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const uint32_t source_tile = ttnn::operations::wavelet::make_symmetric_index_i32(
                                             raw_begin + static_cast<int32_t>(offset), logical_length) /
                                         kTileSide;
            bool found = false;
            for (uint32_t index = 0; index < count; ++index) {
                found = found || tiles[index] == source_tile;
            }
            if (!found && count < capacity) {
                tiles[count++] = source_tile;
            }
        }
        return;
    }
    const SourceAxisTileCollector collector{
        .tiles = tiles,
        .count = count,
        .capacity = capacity,
    };
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kAntireflect) {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const int32_t raw_index = raw_begin + static_cast<int32_t>(offset);
            const ttnn::operations::wavelet::AntireflectIndexI32 extended =
                ttnn::operations::wavelet::make_antireflect_index_i32(raw_index, logical_length);
            collector(extended.source_index);
        }
        // Affine extension may use both endpoint values, but their tile IDs
        // are invariant across this whole macro tile.
        collector(0);
        collector(logical_length - 1U);
    } else if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kSmooth) {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const int32_t raw_index = raw_begin + static_cast<int32_t>(offset);
            const ttnn::operations::wavelet::SmoothIndexI32 extended =
                ttnn::operations::wavelet::make_smooth_index_i32(raw_index, logical_length);
            ttnn::operations::wavelet::visit_smooth_source_indices_i32(extended, collector);
        }
    } else {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const int32_t raw_index = raw_begin + static_cast<int32_t>(offset);
            const ttnn::operations::wavelet::ExtendedIndexI32 extended =
                ttnn::operations::wavelet::make_extended_index_i32<Mode>(raw_index, logical_length);
            ttnn::operations::wavelet::visit_extended_source_indices_i32<Mode>(extended, logical_length, collector);
        }
    }
}

template <bool Interior, ttnn::operations::wavelet::BoundaryMode Mode>
ALWI void collect_source_axis_tiles(
    uint32_t* tiles, uint32_t& count, const uint32_t capacity, const int32_t raw_begin, const uint32_t logical_length) {
    if constexpr (Interior) {
        const uint32_t source_begin = static_cast<uint32_t>(raw_begin);
        const uint32_t source_end = source_begin + 2 * kTileSide - 1;
        for (uint32_t tile = source_begin / kTileSide; tile <= source_end / kTileSide; ++tile) {
            tiles[count++] = tile;
        }
    } else {
        collect_boundary_source_axis_tiles<Mode>(tiles, count, capacity, raw_begin, logical_length);
    }
}

template <bool Interior, ttnn::operations::wavelet::BoundaryMode Mode, typename InputAccessor>
[[nodiscard]] ALWI SplitSourceTiles<Mode> stage_split_source_tiles(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t input_tile_base,
    const int32_t raw_y_begin,
    const int32_t raw_x_begin,
    const uint32_t scratch_addr) {
    SplitSourceTiles<Mode> tiles{};
    collect_source_axis_tiles<Interior, Mode>(
        tiles.rows, tiles.row_count, SplitSourceTiles<Mode>::kAxisCapacity, raw_y_begin, input_height);
    collect_source_axis_tiles<Interior, Mode>(
        tiles.columns, tiles.column_count, SplitSourceTiles<Mode>::kAxisCapacity, raw_x_begin, input_width);

    Noc noc;
    for (uint32_t tile_y = 0; tile_y < tiles.row_count; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tiles.column_count; ++tile_x) {
            const uint32_t source_tile = tiles.rows[tile_y] * input_tile_columns + tiles.columns[tile_x];
            const uint32_t scratch_tile = tile_y * tiles.column_count + tile_x;
            noc.async_read(
                input,
                CoreLocalMem<uint32_t>(scratch_addr + scratch_tile * kTileBytes),
                kTileBytes,
                {.page_id = input_tile_base + source_tile},
                {});
        }
    }
    noc.async_read_barrier();
    return tiles;
}

template <ttnn::operations::wavelet::BoundaryMode Mode>
struct StagedInputColumnReader {
    uint32_t source_y;
    const SplitSourceTiles<Mode>& source_tiles;
    uint32_t scratch_addr;

    LWT_2D_BOUNDARY_FUNCTION float operator()(const uint32_t source_x) const {
        const uint32_t source_tile_y = find_index(source_tiles.rows, source_tiles.row_count, source_y / kTileSide);
        const uint32_t source_tile_x =
            find_index(source_tiles.columns, source_tiles.column_count, source_x / kTileSide);
        const uint32_t scratch_tile = source_tile_y * source_tiles.column_count + source_tile_x;
        const auto* source = reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_tile * kTileBytes);
        return source[tile_element_offset(source_y % kTileSide, source_x % kTileSide)];
    }
};

template <ttnn::operations::wavelet::BoundaryMode Mode>
struct StagedInputRowReader {
    const ttnn::operations::wavelet::ExtendedIndexI32& x_extended;
    uint32_t input_width;
    const SplitSourceTiles<Mode>& source_tiles;
    uint32_t scratch_addr;

    ALWI float operator()(const uint32_t source_y) const {
        return ttnn::operations::wavelet::evaluate_extended_index_i32<Mode>(
            x_extended,
            input_width,
            StagedInputColumnReader<Mode>{
                .source_y = source_y,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
};

struct StagedAntireflectInputRowReader {
    const ttnn::operations::wavelet::AntireflectIndexI32& x_extended;
    uint32_t input_width;
    const SplitSourceTiles<ttnn::operations::wavelet::BoundaryMode::kAntireflect>& source_tiles;
    uint32_t scratch_addr;

    LWT_2D_BOUNDARY_FUNCTION float operator()(const uint32_t source_y) const {
        return ttnn::operations::wavelet::evaluate_antireflect_index_i32(
            x_extended,
            input_width,
            StagedInputColumnReader<ttnn::operations::wavelet::BoundaryMode::kAntireflect>{
                .source_y = source_y,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
};

struct StagedSmoothInputRowReader {
    const ttnn::operations::wavelet::SmoothIndexI32& x_extended;
    const SplitSourceTiles<ttnn::operations::wavelet::BoundaryMode::kSmooth>& source_tiles;
    uint32_t scratch_addr;

    LWT_2D_BOUNDARY_FUNCTION float operator()(const uint32_t source_y) const {
        return ttnn::operations::wavelet::evaluate_smooth_index_i32(
            x_extended,
            StagedInputColumnReader<ttnn::operations::wavelet::BoundaryMode::kSmooth>{
                .source_y = source_y,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
};

template <ttnn::operations::wavelet::BoundaryMode Mode>
[[nodiscard]] ALWI float read_staged_extended_2d(
    const int32_t raw_y,
    const int32_t raw_x,
    const uint32_t input_height,
    const uint32_t input_width,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kAntireflect) {
        const ttnn::operations::wavelet::AntireflectIndexI32 y_extended =
            ttnn::operations::wavelet::make_antireflect_index_i32(raw_y, input_height);
        const ttnn::operations::wavelet::AntireflectIndexI32 x_extended =
            ttnn::operations::wavelet::make_antireflect_index_i32(raw_x, input_width);
        return ttnn::operations::wavelet::evaluate_antireflect_index_i32(
            y_extended,
            input_height,
            StagedAntireflectInputRowReader{
                .x_extended = x_extended,
                .input_width = input_width,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kSmooth) {
        const ttnn::operations::wavelet::SmoothIndexI32 y_extended =
            ttnn::operations::wavelet::make_smooth_index_i32(raw_y, input_height);
        const ttnn::operations::wavelet::SmoothIndexI32 x_extended =
            ttnn::operations::wavelet::make_smooth_index_i32(raw_x, input_width);
        return ttnn::operations::wavelet::evaluate_smooth_index_i32(
            y_extended,
            StagedSmoothInputRowReader{
                .x_extended = x_extended,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kSymmetric) {
        const uint32_t source_y = ttnn::operations::wavelet::make_symmetric_index_i32(raw_y, input_height);
        const uint32_t source_x = ttnn::operations::wavelet::make_symmetric_index_i32(raw_x, input_width);
        return StagedInputColumnReader<Mode>{
            .source_y = source_y,
            .source_tiles = source_tiles,
            .scratch_addr = scratch_addr,
        }(source_x);
    }
    const ttnn::operations::wavelet::ExtendedIndexI32 y_extended =
        ttnn::operations::wavelet::make_extended_index_i32<Mode>(raw_y, input_height);
    const ttnn::operations::wavelet::ExtendedIndexI32 x_extended =
        ttnn::operations::wavelet::make_extended_index_i32<Mode>(raw_x, input_width);
    return ttnn::operations::wavelet::evaluate_extended_index_i32<Mode>(
        y_extended,
        input_height,
        StagedInputRowReader<Mode>{
            .x_extended = x_extended,
            .input_width = input_width,
            .source_tiles = source_tiles,
            .scratch_addr = scratch_addr,
        });
}

#if defined(LWT_2D_COMPACT_BOUNDARY_CODE) || defined(ILWT_2D_COMPACT_BOUNDARY_CODE)
#define LWT_2D_POLYPHASE_TEMPLATE template <bool Interior, ttnn::operations::wavelet::BoundaryMode Mode>
#define LWT_2D_POLYPHASE_FUNCTION __attribute__((noinline))
#define LWT_2D_POLYPHASE_PARITY_PARAMETERS const uint32_t parity_y, const uint32_t parity_x,
#define LWT_2D_POLYPHASE_PARITY_Y parity_y
#define LWT_2D_POLYPHASE_PARITY_X parity_x
#else
#define LWT_2D_POLYPHASE_TEMPLATE \
    template <bool Interior, ttnn::operations::wavelet::BoundaryMode Mode, uint32_t ParityY, uint32_t ParityX>
#define LWT_2D_POLYPHASE_FUNCTION ALWI
#define LWT_2D_POLYPHASE_PARITY_PARAMETERS
#define LWT_2D_POLYPHASE_PARITY_Y ParityY
#define LWT_2D_POLYPHASE_PARITY_X ParityX
#endif

LWT_2D_POLYPHASE_TEMPLATE
LWT_2D_POLYPHASE_FUNCTION void write_polyphase_tile(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t pad_y,
    const uint32_t pad_x,
    LWT_2D_POLYPHASE_PARITY_PARAMETERS const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
    if (!intersects_tile(rectangle, tile_y, tile_x)) {
        return;
    }

    const uint32_t rectangle_y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t rectangle_x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t plane_tile_y = (tile_y - rectangle_y_origin) / kTileSide;
    const uint32_t plane_tile_x = (tile_x - rectangle_x_origin) / kTileSide;
    const uint32_t destination_tile_index = plane_tile_y * plane_tile_columns + plane_tile_x;
    auto* destination = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr + destination_tile_index * kTileBytes);

    const uint32_t y_begin = std::max(rectangle.y_begin, tile_y);
    const uint32_t y_end = std::min(rectangle.y_begin + rectangle.y_length, tile_y + kTileSide);
    const uint32_t x_begin = std::max(rectangle.x_begin, tile_x);
    const uint32_t x_end = std::min(rectangle.x_begin + rectangle.x_length, tile_x + kTileSide);
    for (uint32_t polyphase_y = y_begin; polyphase_y < y_end; ++polyphase_y) {
        const int32_t raw_y = 2 * static_cast<int32_t>(polyphase_y) + static_cast<int32_t>(LWT_2D_POLYPHASE_PARITY_Y) -
                              static_cast<int32_t>(pad_y);
        for (uint32_t polyphase_x = x_begin; polyphase_x < x_end; ++polyphase_x) {
            const int32_t raw_x = 2 * static_cast<int32_t>(polyphase_x) +
                                  static_cast<int32_t>(LWT_2D_POLYPHASE_PARITY_X) - static_cast<int32_t>(pad_x);
            if constexpr (Interior) {
                const uint32_t source_y = static_cast<uint32_t>(raw_y);
                const uint32_t source_x = static_cast<uint32_t>(raw_x);
                const uint32_t source_tile_y =
                    find_index(source_tiles.rows, source_tiles.row_count, source_y / kTileSide);
                const uint32_t source_tile_x =
                    find_index(source_tiles.columns, source_tiles.column_count, source_x / kTileSide);
                const uint32_t scratch_tile = source_tile_y * source_tiles.column_count + source_tile_x;
                const auto* source =
                    reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_tile * kTileBytes);
                destination[tile_element_offset(polyphase_y - tile_y, polyphase_x - tile_x)] =
                    source[tile_element_offset(source_y % kTileSide, source_x % kTileSide)];
            } else {
                destination[tile_element_offset(polyphase_y - tile_y, polyphase_x - tile_x)] =
                    read_staged_extended_2d<Mode>(raw_y, raw_x, input_height, input_width, source_tiles, scratch_addr);
            }
        }
    }
}

#undef LWT_2D_POLYPHASE_TEMPLATE
#undef LWT_2D_POLYPHASE_FUNCTION
#undef LWT_2D_POLYPHASE_PARITY_PARAMETERS
#undef LWT_2D_POLYPHASE_PARITY_Y
#undef LWT_2D_POLYPHASE_PARITY_X

template <bool Interior, ttnn::operations::wavelet::BoundaryMode Mode, uint32_t ParityY, uint32_t ParityX>
ALWI void write_polyphase_tile_dispatch(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
#if defined(LWT_2D_COMPACT_BOUNDARY_CODE) || defined(ILWT_2D_COMPACT_BOUNDARY_CODE)
    write_polyphase_tile<Interior, Mode>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        ParityY,
        ParityX,
        rectangle,
        plane_addr,
        plane_tile_columns,
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
#else
    write_polyphase_tile<Interior, Mode, ParityY, ParityX>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangle,
        plane_addr,
        plane_tile_columns,
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
#endif
}

struct SplitSourceColumn {
    uint32_t scratch_tile_byte_offset;
    uint32_t face_column_offset;
};

[[nodiscard]] ALWI uint32_t split_destination_tile_address(
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x) {
    const uint32_t rectangle_y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t rectangle_x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t plane_tile_y = (tile_y - rectangle_y_origin) / kTileSide;
    const uint32_t plane_tile_x = (tile_x - rectangle_x_origin) / kTileSide;
    return plane_addr + (plane_tile_y * plane_tile_columns + plane_tile_x) * kTileBytes;
}

template <ttnn::operations::wavelet::BoundaryMode Mode>
ALWI void write_full_interior_polyphase_tiles(
    const int32_t raw_y_begin,
    const int32_t raw_x_begin,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
    SplitSourceColumn even_columns[kTileSide];
    SplitSourceColumn odd_columns[kTileSide];
    uint32_t destination_column_offsets[kTileSide];
    const uint32_t first_source_tile_column = source_tiles.columns[0];
    for (uint32_t column = 0; column < kTileSide; ++column) {
        const uint32_t source_even_x = static_cast<uint32_t>(raw_x_begin + 2 * static_cast<int32_t>(column));
        const uint32_t source_odd_x = source_even_x + 1;
        destination_column_offsets[column] = tile_face_column_offset(column);
        even_columns[column] = SplitSourceColumn{
            .scratch_tile_byte_offset = (source_even_x / kTileSide - first_source_tile_column) * kTileBytes,
            .face_column_offset = tile_face_column_offset(source_even_x % kTileSide),
        };
        odd_columns[column] = SplitSourceColumn{
            .scratch_tile_byte_offset = (source_odd_x / kTileSide - first_source_tile_column) * kTileBytes,
            .face_column_offset = tile_face_column_offset(source_odd_x % kTileSide),
        };
    }

    auto* ee = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(rectangles[0], plane_addrs[0], plane_tile_columns[0], tile_y, tile_x));
    auto* eo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(rectangles[1], plane_addrs[1], plane_tile_columns[1], tile_y, tile_x));
    auto* oe = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(rectangles[2], plane_addrs[2], plane_tile_columns[2], tile_y, tile_x));
    auto* oo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(rectangles[3], plane_addrs[3], plane_tile_columns[3], tile_y, tile_x));

    const uint32_t first_source_tile_row = source_tiles.rows[0];
    const uint32_t scratch_tile_row_stride = source_tiles.column_count * kTileBytes;
    for (uint32_t row = 0; row < kTileSide; ++row) {
        const uint32_t source_even_y = static_cast<uint32_t>(raw_y_begin + 2 * static_cast<int32_t>(row));
        const uint32_t source_odd_y = source_even_y + 1;
        const uint32_t source_even_row_base =
            scratch_addr + (source_even_y / kTileSide - first_source_tile_row) * scratch_tile_row_stride;
        const uint32_t source_odd_row_base =
            scratch_addr + (source_odd_y / kTileSide - first_source_tile_row) * scratch_tile_row_stride;
        const uint32_t source_even_row_offset = tile_face_row_offset(source_even_y % kTileSide);
        const uint32_t source_odd_row_offset = tile_face_row_offset(source_odd_y % kTileSide);
        const uint32_t destination_row_offset = tile_face_row_offset(row);

        for (uint32_t column = 0; column < kTileSide; ++column) {
            const SplitSourceColumn even_column = even_columns[column];
            const SplitSourceColumn odd_column = odd_columns[column];
            const auto* even_row_even_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_even_row_base + even_column.scratch_tile_byte_offset);
            const auto* even_row_odd_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_even_row_base + odd_column.scratch_tile_byte_offset);
            const auto* odd_row_even_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_odd_row_base + even_column.scratch_tile_byte_offset);
            const auto* odd_row_odd_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_odd_row_base + odd_column.scratch_tile_byte_offset);
            const uint32_t destination_offset = destination_row_offset + destination_column_offsets[column];
            ee[destination_offset] = even_row_even_column[source_even_row_offset + even_column.face_column_offset];
            eo[destination_offset] = even_row_odd_column[source_even_row_offset + odd_column.face_column_offset];
            oe[destination_offset] = odd_row_even_column[source_odd_row_offset + even_column.face_column_offset];
            oo[destination_offset] = odd_row_odd_column[source_odd_row_offset + odd_column.face_column_offset];
        }
    }
}

template <bool Interior, ttnn::operations::wavelet::BoundaryMode Mode, typename InputAccessor>
ALWI void split_macro_tile(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t input_tile_base,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const uint32_t scratch_addr) {
    const int32_t raw_y_begin = 2 * static_cast<int32_t>(tile_y) - static_cast<int32_t>(pad_y);
    const int32_t raw_x_begin = 2 * static_cast<int32_t>(tile_x) - static_cast<int32_t>(pad_x);
    const SplitSourceTiles<Mode> source_tiles = stage_split_source_tiles<Interior, Mode>(
        input, input_height, input_width, input_tile_columns, input_tile_base, raw_y_begin, raw_x_begin, scratch_addr);

    if constexpr (Interior) {
        bool complete = true;
        for (uint32_t plane = 0; plane < 4; ++plane) {
            complete = complete && covers_tile(rectangles[plane], tile_y, tile_x);
        }
        if (complete) {
            write_full_interior_polyphase_tiles<Mode>(
                raw_y_begin,
                raw_x_begin,
                rectangles,
                plane_addrs,
                plane_tile_columns,
                tile_y,
                tile_x,
                source_tiles,
                scratch_addr);
            return;
        }
    }

    write_polyphase_tile_dispatch<Interior, Mode, 0, 0>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[0],
        plane_addrs[0],
        plane_tile_columns[0],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
    write_polyphase_tile_dispatch<Interior, Mode, 0, 1>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[1],
        plane_addrs[1],
        plane_tile_columns[1],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
    write_polyphase_tile_dispatch<Interior, Mode, 1, 0>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[2],
        plane_addrs[2],
        plane_tile_columns[2],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
    write_polyphase_tile_dispatch<Interior, Mode, 1, 1>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[3],
        plane_addrs[3],
        plane_tile_columns[3],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
}

template <ttnn::operations::wavelet::BoundaryMode Mode, typename InputAccessor>
ALWI void initialize_planes_tiled(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t input_tile_base,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t scratch_addr) {
    uint32_t y_begin = rectangles[0].y_begin;
    uint32_t y_end = rectangles[0].y_begin + rectangles[0].y_length;
    uint32_t x_begin = rectangles[0].x_begin;
    uint32_t x_end = rectangles[0].x_begin + rectangles[0].x_length;
    for (uint32_t plane = 1; plane < 4; ++plane) {
        y_begin = std::min(y_begin, rectangles[plane].y_begin);
        y_end = std::max(y_end, rectangles[plane].y_begin + rectangles[plane].y_length);
        x_begin = std::min(x_begin, rectangles[plane].x_begin);
        x_end = std::max(x_end, rectangles[plane].x_begin + rectangles[plane].x_length);
    }
    for (uint32_t tile_y = aligned_begin(y_begin); tile_y < aligned_end(y_begin, y_end - y_begin);
         tile_y += kTileSide) {
        for (uint32_t tile_x = aligned_begin(x_begin); tile_x < aligned_end(x_begin, x_end - x_begin);
             tile_x += kTileSide) {
            bool active = false;
            for (uint32_t plane = 0; plane < 4; ++plane) {
                active = active || intersects_tile(rectangles[plane], tile_y, tile_x);
            }
            if (!active) {
                continue;
            }
            const int32_t raw_y_begin = 2 * static_cast<int32_t>(tile_y) - static_cast<int32_t>(pad_y);
            const int32_t raw_x_begin = 2 * static_cast<int32_t>(tile_x) - static_cast<int32_t>(pad_x);
            const bool interior =
                raw_y_begin >= 0 && raw_x_begin >= 0 &&
                raw_y_begin + static_cast<int32_t>(2 * kTileSide) <= static_cast<int32_t>(input_height) &&
                raw_x_begin + static_cast<int32_t>(2 * kTileSide) <= static_cast<int32_t>(input_width);
            if (interior) {
                split_macro_tile<true, Mode>(
                    input,
                    input_height,
                    input_width,
                    input_tile_columns,
                    input_tile_base,
                    pad_y,
                    pad_x,
                    rectangles,
                    plane_addrs,
                    plane_tile_columns,
                    tile_y,
                    tile_x,
                    scratch_addr);
            } else {
                split_macro_tile<false, Mode>(
                    input,
                    input_height,
                    input_width,
                    input_tile_columns,
                    input_tile_base,
                    pad_y,
                    pad_x,
                    rectangles,
                    plane_addrs,
                    plane_tile_columns,
                    tile_y,
                    tile_x,
                    scratch_addr);
            }
        }
    }
}

#ifdef ILWT_2D
template <typename BandAccessor>
ALWI void initialize_inverse_band_plane(
    const BandAccessor& band_args,
    const uint32_t band_addr,
    const uint32_t band_height,
    const uint32_t band_width,
    const uint32_t band_tile_columns,
    const uint32_t band_tile_base,
    const int32_t y_internal_offset,
    const int32_t x_internal_offset,
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t scratch_addr,
    const uint32_t zero_tile_addr) {
    const auto band = TensorAccessor(band_args, band_addr, kTileBytes);
    Noc noc;
    UnicastEndpoint local_endpoint;
    const uint32_t y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t y_end = aligned_end(rectangle.y_begin, rectangle.y_length);
    const uint32_t x_end = aligned_end(rectangle.x_begin, rectangle.x_length);

    for (uint32_t tile_y = y_origin; tile_y < y_end; tile_y += kTileSide) {
        for (uint32_t tile_x = x_origin; tile_x < x_end; tile_x += kTileSide) {
            const uint32_t destination_tile =
                ((tile_y - y_origin) / kTileSide) * plane_tile_columns + (tile_x - x_origin) / kTileSide;
            const uint32_t destination_addr = plane_addr + destination_tile * kTileBytes;
            const int32_t full_canonical_y = static_cast<int32_t>(tile_y) - y_internal_offset;
            const int32_t full_canonical_x = static_cast<int32_t>(tile_x) - x_internal_offset;
            const bool exact_full_tile =
                tile_y >= rectangle.y_begin && tile_y + kTileSide <= rectangle.y_begin + rectangle.y_length &&
                tile_x >= rectangle.x_begin && tile_x + kTileSide <= rectangle.x_begin + rectangle.x_length &&
                full_canonical_y >= 0 && full_canonical_x >= 0 &&
                full_canonical_y % static_cast<int32_t>(kTileSide) == 0 &&
                full_canonical_x % static_cast<int32_t>(kTileSide) == 0 &&
                full_canonical_y + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(band_height) &&
                full_canonical_x + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(band_width);
            if (exact_full_tile) {
                const uint32_t source_tile = (static_cast<uint32_t>(full_canonical_y) / kTileSide) * band_tile_columns +
                                             static_cast<uint32_t>(full_canonical_x) / kTileSide;
                noc.async_read(
                    band,
                    CoreLocalMem<uint32_t>(destination_addr),
                    kTileBytes,
                    {.page_id = band_tile_base + source_tile},
                    {});
                continue;
            }
            noc.async_read(
                local_endpoint,
                CoreLocalMem<uint32_t>(destination_addr),
                kTileBytes,
                ttnn::operations::wavelet::kernels::primitives::local_noc_source(noc, zero_tile_addr),
                {});
            noc.async_read_barrier();

            const uint32_t internal_y_begin = std::max(tile_y, rectangle.y_begin);
            const uint32_t internal_y_end = std::min(tile_y + kTileSide, rectangle.y_begin + rectangle.y_length);
            const uint32_t internal_x_begin = std::max(tile_x, rectangle.x_begin);
            const uint32_t internal_x_end = std::min(tile_x + kTileSide, rectangle.x_begin + rectangle.x_length);
            if (internal_y_begin == internal_y_end || internal_x_begin == internal_x_end) {
                continue;
            }

            const int32_t canonical_y_begin = static_cast<int32_t>(internal_y_begin) - y_internal_offset;
            const int32_t canonical_y_end = static_cast<int32_t>(internal_y_end) - y_internal_offset;
            const int32_t canonical_x_begin = static_cast<int32_t>(internal_x_begin) - x_internal_offset;
            const int32_t canonical_x_end = static_cast<int32_t>(internal_x_end) - x_internal_offset;
            ASSERT(canonical_y_begin >= 0 && canonical_x_begin >= 0);
            ASSERT(canonical_y_end <= static_cast<int32_t>(band_height));
            ASSERT(canonical_x_end <= static_cast<int32_t>(band_width));

            const uint32_t source_tile_y_begin = static_cast<uint32_t>(canonical_y_begin) / kTileSide;
            const uint32_t source_tile_y_end = (static_cast<uint32_t>(canonical_y_end - 1) / kTileSide) + 1;
            const uint32_t source_tile_x_begin = static_cast<uint32_t>(canonical_x_begin) / kTileSide;
            const uint32_t source_tile_x_end = (static_cast<uint32_t>(canonical_x_end - 1) / kTileSide) + 1;
            const uint32_t source_tile_rows = source_tile_y_end - source_tile_y_begin;
            const uint32_t source_tile_columns = source_tile_x_end - source_tile_x_begin;
            ASSERT(source_tile_rows <= ttnn::operations::wavelet::device_protocol::kLwt2DSplitScratchTileRows);
            ASSERT(source_tile_columns <= ttnn::operations::wavelet::device_protocol::kLwt2DSplitScratchTileColumns);

            for (uint32_t source_tile_y = 0; source_tile_y < source_tile_rows; ++source_tile_y) {
                for (uint32_t source_tile_x = 0; source_tile_x < source_tile_columns; ++source_tile_x) {
                    const uint32_t source_tile =
                        (source_tile_y_begin + source_tile_y) * band_tile_columns + source_tile_x_begin + source_tile_x;
                    const uint32_t scratch_tile = source_tile_y * source_tile_columns + source_tile_x;
                    noc.async_read(
                        band,
                        CoreLocalMem<uint32_t>(scratch_addr + scratch_tile * kTileBytes),
                        kTileBytes,
                        {.page_id = band_tile_base + source_tile},
                        {});
                }
            }
            noc.async_read_barrier();

            auto* destination = reinterpret_cast<volatile tt_l1_ptr float*>(destination_addr);
            for (uint32_t internal_y = internal_y_begin; internal_y < internal_y_end; ++internal_y) {
                const uint32_t canonical_y =
                    static_cast<uint32_t>(static_cast<int32_t>(internal_y) - y_internal_offset);
                const uint32_t source_tile_y = canonical_y / kTileSide - source_tile_y_begin;
                for (uint32_t internal_x = internal_x_begin; internal_x < internal_x_end; ++internal_x) {
                    const uint32_t canonical_x =
                        static_cast<uint32_t>(static_cast<int32_t>(internal_x) - x_internal_offset);
                    const uint32_t source_tile_x = canonical_x / kTileSide - source_tile_x_begin;
                    const uint32_t scratch_tile = source_tile_y * source_tile_columns + source_tile_x;
                    const auto* source =
                        reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_tile * kTileBytes);
                    destination[tile_element_offset(internal_y - tile_y, internal_x - tile_x)] =
                        source[tile_element_offset(canonical_y % kTileSide, canonical_x % kTileSide)];
                }
            }
        }
    }
    noc.async_read_barrier();
}

template <typename LlAccessor, typename LhAccessor, typename HlAccessor, typename HhAccessor>
ALWI void initialize_inverse_band_planes(
    const LlAccessor& ll_args,
    const LhAccessor& lh_args,
    const HlAccessor& hl_args,
    const HhAccessor& hh_args,
    const uint32_t* band_addrs,
    const uint32_t band_height,
    const uint32_t band_width,
    const uint32_t band_tile_columns,
    const uint32_t band_tile_base,
    const int32_t* y_internal_offsets,
    const int32_t* x_internal_offsets,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t scratch_addr,
    const uint32_t zero_tile_addr) {
    // LL, LH, HL, HH map to (low-y,low-x), (low-y,high-x),
    // (high-y,low-x), and (high-y,high-x).
    initialize_inverse_band_plane(
        ll_args,
        band_addrs[0],
        band_height,
        band_width,
        band_tile_columns,
        band_tile_base,
        y_internal_offsets[0],
        x_internal_offsets[0],
        rectangles[0],
        plane_addrs[0],
        plane_tile_columns[0],
        scratch_addr,
        zero_tile_addr);
    initialize_inverse_band_plane(
        lh_args,
        band_addrs[1],
        band_height,
        band_width,
        band_tile_columns,
        band_tile_base,
        y_internal_offsets[0],
        x_internal_offsets[1],
        rectangles[1],
        plane_addrs[1],
        plane_tile_columns[1],
        scratch_addr,
        zero_tile_addr);
    initialize_inverse_band_plane(
        hl_args,
        band_addrs[2],
        band_height,
        band_width,
        band_tile_columns,
        band_tile_base,
        y_internal_offsets[1],
        x_internal_offsets[0],
        rectangles[2],
        plane_addrs[2],
        plane_tile_columns[2],
        scratch_addr,
        zero_tile_addr);
    initialize_inverse_band_plane(
        hh_args,
        band_addrs[3],
        band_height,
        band_width,
        band_tile_columns,
        band_tile_base,
        y_internal_offsets[1],
        x_internal_offsets[1],
        rectangles[3],
        plane_addrs[3],
        plane_tile_columns[3],
        scratch_addr,
        zero_tile_addr);
}
#endif

[[nodiscard]] ALWI float read_plane(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t y,
    const int32_t x) {
    if (!contains(stored, y, x)) {
        return 0.0F;
    }
    const uint32_t local_y = static_cast<uint32_t>(y) - aligned_begin(stored.y_begin);
    const uint32_t local_x = static_cast<uint32_t>(x) - aligned_begin(stored.x_begin);
    const auto* source = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr);
    return source[tiled_element_offset(local_y, local_x, plane_tile_columns)];
}

enum class RouteTileClass : uint32_t {
    kExact,
    kOneAxisShifted,
    kTwoAxisShifted,
    kPartial,
    kEmpty,
};

ALWI void reserve_tile(const uint32_t cb) {
    CircularBuffer buffer(cb);
    buffer.reserve_back(1);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(buffer.get_write_ptr());
    for (uint32_t element = 0; element < kTileElements; ++element) {
        tile[element] = 0.0F;
    }
}

enum class StageTileResult : uint32_t {
    kExactPending,
    kBoundedPending,
    kCompleted,
};

[[nodiscard]] ALWI bool requested_tile_inside(
    const Rect& stored, const int32_t requested_y, const int32_t requested_x) {
    return requested_y >= static_cast<int32_t>(stored.y_begin) && requested_x >= static_cast<int32_t>(stored.x_begin) &&
           requested_y + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(stored.y_begin + stored.y_length) &&
           requested_x + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(stored.x_begin + stored.x_length);
}

[[nodiscard]] ALWI RouteTileClass
classify_route_tile(const Rect& stored, const int32_t requested_y, const int32_t requested_x) {
    if (!requested_tile_inside(stored, requested_y, requested_x)) {
        const int32_t requested_y_end = requested_y + static_cast<int32_t>(kTileSide);
        const int32_t requested_x_end = requested_x + static_cast<int32_t>(kTileSide);
        const int32_t stored_y_end = static_cast<int32_t>(stored.y_begin + stored.y_length);
        const int32_t stored_x_end = static_cast<int32_t>(stored.x_begin + stored.x_length);
        const bool intersects = requested_y < stored_y_end && requested_y_end > static_cast<int32_t>(stored.y_begin) &&
                                requested_x < stored_x_end && requested_x_end > static_cast<int32_t>(stored.x_begin);
        return intersects ? RouteTileClass::kPartial : RouteTileClass::kEmpty;
    }
    const bool y_aligned =
        (requested_y - static_cast<int32_t>(aligned_begin(stored.y_begin))) % static_cast<int32_t>(kTileSide) == 0;
    const bool x_aligned =
        (requested_x - static_cast<int32_t>(aligned_begin(stored.x_begin))) % static_cast<int32_t>(kTileSide) == 0;
    if (y_aligned && x_aligned) {
        return RouteTileClass::kExact;
    }
    if (y_aligned != x_aligned) {
        return RouteTileClass::kOneAxisShifted;
    }
    return RouteTileClass::kTwoAxisShifted;
}

[[nodiscard]] ALWI uint32_t route_plane_tile_addr(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    const uint32_t local_y = static_cast<uint32_t>(requested_y) - aligned_begin(stored.y_begin);
    const uint32_t local_x = static_cast<uint32_t>(requested_x) - aligned_begin(stored.x_begin);
    const uint32_t tile_index = (local_y / kTileSide) * plane_tile_columns + local_x / kTileSide;
    return plane_addr + tile_index * kTileBytes;
}

ALWI void copy_contiguous_words(
    volatile tt_l1_ptr uint32_t* destination, const volatile tt_l1_ptr uint32_t* source, const uint32_t word_count) {
    for (uint32_t word = 0; word < word_count; ++word) {
        destination[word] = source[word];
    }
}

ALWI void assemble_one_axis_shifted_tile(
    const uint32_t destination_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
    const auto* source = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(plane_addr);
    const uint32_t stored_y_origin = aligned_begin(stored.y_begin);
    const uint32_t stored_x_origin = aligned_begin(stored.x_begin);
    for (uint32_t row = 0; row < kTileSide; ++row) {
        uint32_t column = 0;
        while (column < kTileSide) {
            const uint32_t source_y = static_cast<uint32_t>(requested_y) + row - stored_y_origin;
            const uint32_t source_x = static_cast<uint32_t>(requested_x) + column - stored_x_origin;
            const uint32_t count = std::min(
                kTileSide - column, std::min(kFaceSide - source_x % kFaceSide, kFaceSide - column % kFaceSide));
            const uint32_t source_offset = tiled_element_offset(source_y, source_x, plane_tile_columns);
            const uint32_t destination_offset = tile_element_offset(row, column);
            copy_contiguous_words(destination + destination_offset, source + source_offset, count);
            column += count;
        }
    }
}

__attribute__((noinline)) void assemble_bounded_tile(
    const uint32_t destination_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    const int32_t valid_y_begin = std::max(requested_y, static_cast<int32_t>(stored.y_begin));
    const int32_t valid_y_end =
        std::min(requested_y + static_cast<int32_t>(kTileSide), static_cast<int32_t>(stored.y_begin + stored.y_length));
    const int32_t valid_x_begin = std::max(requested_x, static_cast<int32_t>(stored.x_begin));
    const int32_t valid_x_end =
        std::min(requested_x + static_cast<int32_t>(kTileSide), static_cast<int32_t>(stored.x_begin + stored.x_length));
    if (valid_y_begin >= valid_y_end || valid_x_begin >= valid_x_end) {
        return;
    }

    auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
    const auto* source = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(plane_addr);
    const uint32_t stored_y_origin = aligned_begin(stored.y_begin);
    const uint32_t stored_x_origin = aligned_begin(stored.x_begin);
    const uint32_t destination_row_begin = static_cast<uint32_t>(valid_y_begin - requested_y);
    const uint32_t destination_column_begin = static_cast<uint32_t>(valid_x_begin - requested_x);
    const uint32_t valid_width = static_cast<uint32_t>(valid_x_end - valid_x_begin);

    for (uint32_t destination_row = destination_row_begin;
         destination_row < destination_row_begin + static_cast<uint32_t>(valid_y_end - valid_y_begin);
         ++destination_row) {
        const uint32_t source_y = static_cast<uint32_t>(valid_y_begin - static_cast<int32_t>(stored_y_origin)) +
                                  destination_row - destination_row_begin;
        uint32_t copied = 0;
        while (copied < valid_width) {
            const uint32_t destination_column = destination_column_begin + copied;
            const uint32_t source_x =
                static_cast<uint32_t>(valid_x_begin - static_cast<int32_t>(stored_x_origin)) + copied;
            const uint32_t count = std::min(
                valid_width - copied,
                std::min(kFaceSide - source_x % kFaceSide, kFaceSide - destination_column % kFaceSide));
            copy_contiguous_words(
                destination + tile_element_offset(destination_row, destination_column),
                source + tiled_element_offset(source_y, source_x, plane_tile_columns),
                count);
            copied += count;
        }
    }
}

[[nodiscard]] __attribute__((noinline)) StageTileResult stage_optimized_tile(
    const uint32_t cb,
    const uint32_t zero_tile_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    const RouteTileClass tile_class = classify_route_tile(stored, requested_y, requested_x);
    CircularBuffer buffer(cb);
    Noc noc;
    UnicastEndpoint local_endpoint;
    buffer.reserve_back(1);
    const uint32_t destination_addr = buffer.get_write_ptr();
    if (tile_class == RouteTileClass::kExact) {
        const uint32_t source_addr =
            route_plane_tile_addr(plane_addr, plane_tile_columns, stored, requested_y, requested_x);
        noc.async_read(
            local_endpoint,
            CoreLocalMem<uint32_t>(destination_addr),
            kTileBytes,
            ttnn::operations::wavelet::kernels::primitives::local_noc_source(noc, source_addr),
            {});
        return StageTileResult::kExactPending;
    }

    if (tile_class == RouteTileClass::kPartial || tile_class == RouteTileClass::kEmpty) {
        noc.async_read(
            local_endpoint,
            CoreLocalMem<uint32_t>(destination_addr),
            kTileBytes,
            ttnn::operations::wavelet::kernels::primitives::local_noc_source(noc, zero_tile_addr),
            {});
        return StageTileResult::kBoundedPending;
    }

    if (tile_class == RouteTileClass::kOneAxisShifted) {
        assemble_one_axis_shifted_tile(
            destination_addr, plane_addr, plane_tile_columns, stored, requested_y, requested_x);
    } else {
        assemble_bounded_tile(destination_addr, plane_addr, plane_tile_columns, stored, requested_y, requested_x);
    }
    buffer.push_back(1);
    return StageTileResult::kCompleted;
}

__attribute__((noinline)) void finish_pending_tile(
    const StageTileResult result,
    const uint32_t cb,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    if (result == StageTileResult::kCompleted) {
        return;
    }
    CircularBuffer buffer(cb);
    const uint32_t destination_addr = buffer.get_write_ptr();
    if (result == StageTileResult::kBoundedPending) {
        assemble_bounded_tile(destination_addr, plane_addr, plane_tile_columns, stored, requested_y, requested_x);
    }
    buffer.push_back(1);
}

[[nodiscard]] ALWI int32_t base_requested_y(const Rect& source, const Rect& output, const uint32_t output_tile_y) {
    const int32_t output_y_origin = static_cast<int32_t>(aligned_begin(output.y_begin) + output_tile_y * kTileSide);
    return static_cast<int32_t>(source.y_begin) + output_y_origin - static_cast<int32_t>(output.y_begin);
}

[[nodiscard]] ALWI int32_t base_requested_x(const Rect& source, const Rect& output, const uint32_t output_tile_x) {
    const int32_t output_x_origin = static_cast<int32_t>(aligned_begin(output.x_begin) + output_tile_x * kTileSide);
    return static_cast<int32_t>(source.x_begin) + output_x_origin - static_cast<int32_t>(output.x_begin);
}

ALWI void stencil_requested_origin(
    const bool vertical,
    const uint32_t source_tile_index,
    const uint32_t coefficient_count,
    const Rect& source,
    const Rect& output,
    const uint32_t output_tile_y,
    const uint32_t output_tile_x,
    int32_t& requested_y,
    int32_t& requested_x) {
    requested_y = base_requested_y(source, output, output_tile_y);
    requested_x = base_requested_x(source, output, output_tile_x);
    if (vertical) {
        requested_y += static_cast<int32_t>(source_tile_index * kTileSide);
    } else {
        requested_x +=
            static_cast<int32_t>(source_tile_index * kTileSide) - static_cast<int32_t>(17 - coefficient_count);
    }
}

}  // namespace

void kernel_main() {
#ifdef ILWT_2D
    uint32_t band_addrs[ttnn::operations::wavelet::device_protocol::kLwt2DBandCount];
    for (uint32_t band = 0; band < ttnn::operations::wavelet::device_protocol::kLwt2DBandCount; ++band) {
        band_addrs[band] = get_arg_val<uint32_t>(band);
    }
    const uint32_t input_height = get_arg_val<uint32_t>(4);
    const uint32_t input_width = get_arg_val<uint32_t>(5);
    const uint32_t input_tile_columns = get_arg_val<uint32_t>(6);
    const int32_t y_internal_offsets[2] = {
        static_cast<int32_t>(get_arg_val<uint32_t>(7)),
        static_cast<int32_t>(get_arg_val<uint32_t>(8)),
    };
    const int32_t x_internal_offsets[2] = {
        static_cast<int32_t>(get_arg_val<uint32_t>(9)),
        static_cast<int32_t>(get_arg_val<uint32_t>(10)),
    };
    constexpr uint32_t plane_arg_base = 11;
#else
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_height = get_arg_val<uint32_t>(1);
    const uint32_t input_width = get_arg_val<uint32_t>(2);
    const uint32_t input_tile_columns = get_arg_val<uint32_t>(3);
    const uint32_t pad_y = get_arg_val<uint32_t>(4);
    const uint32_t pad_x = get_arg_val<uint32_t>(5);
    constexpr uint32_t plane_arg_base = 6;
#endif
    uint32_t plane_addrs[ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount];
    uint32_t plane_tile_columns[ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount];
    const uint32_t workspace_base =
        CircularBuffer(ttnn::operations::wavelet::device_protocol::kLwt2DWorkspaceCb).get_write_ptr();
    for (uint32_t slot = 0; slot < ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount; ++slot) {
        plane_addrs[slot] = workspace_base + get_arg_val<uint32_t>(plane_arg_base + slot);
        plane_tile_columns[slot] =
            get_arg_val<uint32_t>(plane_arg_base + ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount + slot);
    }
    constexpr uint32_t plane_arg_count = 2 * ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount;
    const uint32_t chunk_config_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count);
    const uint32_t route_config_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 1);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 2);
    const uint32_t chunk_count = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 3);
    const uint32_t route_count = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 4);
    const uint32_t chunks_per_sample = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 5);
    const uint32_t input_tiles_per_sample = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 6);

    constexpr uint32_t cb_source0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_source1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(3);
    constexpr uint32_t cb_chunk_config = get_compile_time_arg_val(4);
    constexpr uint32_t cb_noc_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t cb_route_zero = get_compile_time_arg_val(6);
#ifdef ILWT_2D
    constexpr auto ll_args = TensorAccessorArgs<7>();
    constexpr auto lh_args = TensorAccessorArgs<ll_args.next_compile_time_args_offset()>();
    constexpr auto hl_args = TensorAccessorArgs<lh_args.next_compile_time_args_offset()>();
    constexpr auto hh_args = TensorAccessorArgs<hl_args.next_compile_time_args_offset()>();
    constexpr auto chunk_args = TensorAccessorArgs<hh_args.next_compile_time_args_offset()>();
#else
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto chunk_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
#endif
    constexpr auto route_args = TensorAccessorArgs<chunk_args.next_compile_time_args_offset()>();
    constexpr uint32_t boundary_mode_arg_offset = route_args.next_compile_time_args_offset();
    constexpr auto boundary_mode =
        static_cast<ttnn::operations::wavelet::BoundaryMode>(get_compile_time_arg_val(boundary_mode_arg_offset));
    constexpr uint32_t split_scratch_bytes = get_compile_time_arg_val(boundary_mode_arg_offset + 1);
    static_assert(
        ttnn::operations::wavelet::is_supported_lwt_boundary_mode(boundary_mode),
        "Unsupported 2D signal-extension mode");
#ifndef ILWT_2D
    const auto input = TensorAccessor(input_args, input_addr, kTileBytes);
#endif
    CircularBuffer sync_buffer(cb_sync);
    Noc noc;
    // This CB is an L1 allocation only
    const uint32_t zero_tile_addr = CircularBuffer(cb_route_zero).get_write_ptr();
    auto* zero_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_tile_addr);
    for (uint32_t word = 0; word < kTileElements; ++word) {
        zero_tile[word] = 0;
    }
    const uint32_t noc_scratch_addr = CircularBuffer(cb_noc_scratch).get_write_ptr();
    constexpr uint32_t reader_config_capacity = split_scratch_bytes / 2;
    const uint32_t reader_config_addr = noc_scratch_addr;

    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        const uint32_t global_work_item = chunk_begin + local_chunk;
        const uint32_t batch_index = global_work_item / chunks_per_sample;
        const uint32_t global_chunk = global_work_item - batch_index * chunks_per_sample;
        const uint32_t input_tile_base = batch_index * input_tiles_per_sample;
        uint32_t chunk_words[ttnn::operations::wavelet::device_protocol::kLwt2DChunkConfigWordCount];
        load_config_page(
            chunk_args,
            chunk_config_addr,
            ttnn::operations::wavelet::device_protocol::kLwt2DChunkConfigPageBytes,
            global_chunk,
            cb_chunk_config,
            chunk_words,
            ttnn::operations::wavelet::device_protocol::kLwt2DChunkConfigWordCount);

        const ConfigWords chunk_config{chunk_words};
        Rect stored[ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount];
        stored[0] = Rect{chunk_config, ttnn::operations::wavelet::device_protocol::kLwt2DInitialEe};
        stored[1] = Rect{chunk_config, ttnn::operations::wavelet::device_protocol::kLwt2DInitialEo};
        stored[2] = Rect{chunk_config, ttnn::operations::wavelet::device_protocol::kLwt2DInitialOe};
        stored[3] = Rect{chunk_config, ttnn::operations::wavelet::device_protocol::kLwt2DInitialOo};
        stored[4] = Rect{};

#ifdef ILWT_2D
        initialize_inverse_band_planes(
            ll_args,
            lh_args,
            hl_args,
            hh_args,
            band_addrs,
            input_height,
            input_width,
            input_tile_columns,
            input_tile_base,
            y_internal_offsets,
            x_internal_offsets,
            stored,
            plane_addrs,
            plane_tile_columns,
            noc_scratch_addr,
            zero_tile_addr);
#else
        initialize_planes_tiled<boundary_mode>(
            input,
            input_height,
            input_width,
            input_tile_columns,
            input_tile_base,
            pad_y,
            pad_x,
            stored,
            plane_addrs,
            plane_tile_columns,
            noc_scratch_addr);
#endif
        for (uint32_t word = 0; word < kTileElements; ++word) {
            zero_tile[word] = 0;
        }
        ASSERT(
            route_count * ttnn::operations::wavelet::device_protocol::kLwt2DRouteConfigPageBytes <=
            reader_config_capacity);
        preload_config_pages(
            route_args,
            route_config_addr,
            ttnn::operations::wavelet::device_protocol::kLwt2DRouteConfigPageBytes,
            global_chunk * route_count,
            route_count,
            reader_config_addr);
        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
            const auto* route_words = reinterpret_cast<const uint32_t*>(
                reader_config_addr +
                route_index * ttnn::operations::wavelet::device_protocol::kLwt2DRouteConfigPageBytes);
            const ConfigWords route_config{route_words};
            const uint32_t flags = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteFlags];
            if ((flags & ttnn::operations::wavelet::device_protocol::kLwt2DRouteFlagMetadataOnly) != 0) {
                continue;
            }
            const bool vertical = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteAxis] == 0;
            const uint32_t source_slot = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteSourceSlot];
            const uint32_t base_slot = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteBaseSlot];
            const uint32_t output_slot = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteOutputSlot];
            const Rect source{route_config, ttnn::operations::wavelet::device_protocol::kLwt2DRouteSourceRect};
            const Rect base{route_config, ttnn::operations::wavelet::device_protocol::kLwt2DRouteBaseRect};
            const Rect output{route_config, ttnn::operations::wavelet::device_protocol::kLwt2DRouteOutputRect};
            const uint32_t output_tile_rows =
                (aligned_end(output.y_begin, output.y_length) - aligned_begin(output.y_begin)) / kTileSide;
            const uint32_t output_tile_columns =
                (aligned_end(output.x_begin, output.x_length) - aligned_begin(output.x_begin)) / kTileSide;
            const bool scale = (flags & ttnn::operations::wavelet::device_protocol::kLwt2DRouteFlagScale) != 0;
            const uint32_t coefficient_count =
                scale ? 1 : (vertical ? source.y_length - output.y_length + 1 : source.x_length - output.x_length + 1);

            for (uint32_t tile_y = 0; tile_y < output_tile_rows; ++tile_y) {
                for (uint32_t tile_x = 0; tile_x < output_tile_columns; ++tile_x) {
                    int32_t source0_requested_y = 0;
                    int32_t source0_requested_x = 0;
                    int32_t source1_requested_y = 0;
                    int32_t source1_requested_x = 0;
                    int32_t base_requested_tile_y = 0;
                    int32_t base_requested_tile_x = 0;
                    StageTileResult source0_result = StageTileResult::kCompleted;
                    StageTileResult source1_result = StageTileResult::kCompleted;
                    StageTileResult base_result = StageTileResult::kCompleted;
                    if (scale) {
                        const int32_t requested_y = base_requested_y(source, output, tile_y);
                        const int32_t requested_x = base_requested_x(source, output, tile_x);
                        source0_requested_y = requested_y;
                        source0_requested_x = requested_x;
                        source0_result = stage_optimized_tile(
                            cb_source0,
                            zero_tile_addr,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            requested_y,
                            requested_x);
                    } else {
                        int32_t requested_y = 0;
                        int32_t requested_x = 0;
                        stencil_requested_origin(
                            vertical, 0, coefficient_count, source, output, tile_y, tile_x, requested_y, requested_x);
                        source0_requested_y = requested_y;
                        source0_requested_x = requested_x;
                        source0_result = stage_optimized_tile(
                            cb_source0,
                            zero_tile_addr,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            requested_y,
                            requested_x);
                        stencil_requested_origin(
                            vertical, 1, coefficient_count, source, output, tile_y, tile_x, requested_y, requested_x);
                        source1_requested_y = requested_y;
                        source1_requested_x = requested_x;
                        source1_result = stage_optimized_tile(
                            cb_source1,
                            zero_tile_addr,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            requested_y,
                            requested_x);
                        requested_y = base_requested_y(base, output, tile_y);
                        requested_x = base_requested_x(base, output, tile_x);
                        base_requested_tile_y = requested_y;
                        base_requested_tile_x = requested_x;
                        base_result = stage_optimized_tile(
                            cb_base,
                            zero_tile_addr,
                            plane_addrs[base_slot],
                            plane_tile_columns[base_slot],
                            stored[base_slot],
                            requested_y,
                            requested_x);
                    }
                    if (source0_result != StageTileResult::kCompleted ||
                        source1_result != StageTileResult::kCompleted || base_result != StageTileResult::kCompleted) {
                        noc.async_read_barrier();
                    }
                    finish_pending_tile(
                        source0_result,
                        cb_source0,
                        plane_addrs[source_slot],
                        plane_tile_columns[source_slot],
                        stored[source_slot],
                        source0_requested_y,
                        source0_requested_x);
                    finish_pending_tile(
                        source1_result,
                        cb_source1,
                        plane_addrs[source_slot],
                        plane_tile_columns[source_slot],
                        stored[source_slot],
                        source1_requested_y,
                        source1_requested_x);
                    finish_pending_tile(
                        base_result,
                        cb_base,
                        plane_addrs[base_slot],
                        plane_tile_columns[base_slot],
                        stored[base_slot],
                        base_requested_tile_y,
                        base_requested_tile_x);
                }
            }
            sync_buffer.wait_front(1);
            sync_buffer.pop_front(1);
            stored[output_slot] = output;
        }
        sync_buffer.wait_front(1);
        sync_buffer.pop_front(1);
    }
}
