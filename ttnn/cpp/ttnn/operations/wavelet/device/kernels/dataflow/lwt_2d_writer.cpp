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
#include "ttnn/operations/wavelet/device/protocol/lwt_2d_config.hpp"

namespace {

using ttnn::operations::wavelet::kernels::primitives::ConfigWords;
using ttnn::operations::wavelet::kernels::primitives::kFaceSide;
using ttnn::operations::wavelet::kernels::primitives::kTileBytes;
using ttnn::operations::wavelet::kernels::primitives::kTileSide;
using ttnn::operations::wavelet::kernels::primitives::load_config_page;
using ttnn::operations::wavelet::kernels::primitives::preload_config_pages;
using ttnn::operations::wavelet::kernels::primitives::tile_element_offset;
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

[[nodiscard]] ALWI uint32_t aligned_begin(const uint32_t value) { return (value / kTileSide) * kTileSide; }

[[nodiscard]] ALWI uint32_t aligned_end(const uint32_t begin, const uint32_t length) {
    return ((begin + length + kTileSide - 1) / kTileSide) * kTileSide;
}

ALWI void write_local_output(
    const uint32_t cb_output, const uint32_t plane_addr, const uint32_t plane_tile_columns, const Rect& output) {
    CircularBuffer output_buffer(cb_output);
    Noc noc;
    UnicastEndpoint local_endpoint;
    const uint32_t tile_rows =
        (aligned_end(output.y_begin, output.y_length) - aligned_begin(output.y_begin)) / kTileSide;
    const uint32_t tile_columns =
        (aligned_end(output.x_begin, output.x_length) - aligned_begin(output.x_begin)) / kTileSide;
    const uint32_t tile_count = tile_rows * tile_columns;
    for (uint32_t first_tile = 0; first_tile < tile_count;) {
        const uint32_t read_ptr = output_buffer.get_read_ptr();
        const uint32_t fifo_limit = get_local_cb_interface(cb_output).fifo_limit;
        const uint32_t batch = first_tile + 1 < tile_count && read_ptr + 2 * kTileBytes <= fifo_limit ? 2U : 1U;
        output_buffer.wait_front(batch);
        for (uint32_t tile_in_batch = 0; tile_in_batch < batch; ++tile_in_batch) {
            const uint32_t flat_tile = first_tile + tile_in_batch;
            const uint32_t tile_y = flat_tile / tile_columns;
            const uint32_t tile_x = flat_tile % tile_columns;
            const uint32_t destination_addr = plane_addr + (tile_y * plane_tile_columns + tile_x) * kTileBytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(read_ptr + tile_in_batch * kTileBytes),
                local_endpoint,
                kTileBytes,
                {},
                ttnn::operations::wavelet::kernels::primitives::local_noc_destination(noc, destination_addr));
        }
        noc.async_write_barrier();
        output_buffer.pop_front(batch);
        first_tile += batch;
    }
}

template <typename OutputAccessor>
ALWI void write_band_fragmented(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t output_tile_base,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& source,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    const uint32_t noc_scratch_addr) {
    const auto output = TensorAccessor(output_args, output_addr, kTileBytes);
    Noc noc;
    const uint32_t source_y_origin = aligned_begin(source.y_begin);
    const uint32_t source_x_origin = aligned_begin(source.x_begin);
    for (uint32_t local_y = 0; local_y < final_y_length; ++local_y) {
        uint32_t local_x = 0;
        while (local_x < final_x_length) {
            const uint32_t source_y = source.y_begin + local_y - source_y_origin;
            const uint32_t source_x = source.x_begin + local_x - source_x_origin;
            const uint32_t destination_y = final_y_begin + local_y;
            const uint32_t destination_x = final_x_begin + local_x;
            const uint32_t count = std::min(
                final_x_length - local_x,
                std::min(kFaceSide - source_x % kFaceSide, kFaceSide - destination_x % kFaceSide));
            const uint32_t source_offset = tiled_element_offset(source_y, source_x, plane_tile_columns) * sizeof(float);
            const uint32_t destination_tile =
                (destination_y / kTileSide) * output_tile_columns + destination_x / kTileSide;
            const uint32_t destination_offset =
                tile_element_offset(destination_y % kTileSide, destination_x % kTileSide) * sizeof(float);
            const uint32_t scratch_lane = destination_offset & 63U;
            auto* staged = reinterpret_cast<volatile tt_l1_ptr float*>(noc_scratch_addr + scratch_lane);
            const auto* source_values = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr + source_offset);
            for (uint32_t value = 0; value < count; ++value) {
                staged[value] = source_values[value];
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(noc_scratch_addr + scratch_lane),
                output,
                count * sizeof(float),
                {},
                {.page_id = output_tile_base + destination_tile, .offset_bytes = destination_offset});
            noc.async_write_barrier();
            local_x += count;
        }
    }
}

template <typename OutputAccessor>
[[nodiscard]] ALWI bool write_band_full_tiles(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t output_tile_base,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& source,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length) {
    const bool exact = final_y_begin % kTileSide == 0 && final_x_begin % kTileSide == 0 &&
                       final_y_length % kTileSide == 0 && final_x_length % kTileSide == 0 &&
                       source.y_begin % kTileSide == 0 && source.x_begin % kTileSide == 0 &&
                       source.y_length == final_y_length && source.x_length == final_x_length;
    if (!exact) {
        return false;
    }

    constexpr uint32_t kWriteBatchTiles = 16;
    const auto output = TensorAccessor(output_args, output_addr, kTileBytes);
    Noc noc;
    const uint32_t tile_rows = final_y_length / kTileSide;
    const uint32_t tile_columns = final_x_length / kTileSide;
    uint32_t outstanding = 0;
    for (uint32_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const uint32_t source_tile = tile_y * plane_tile_columns + tile_x;
            const uint32_t destination_tile =
                (final_y_begin / kTileSide + tile_y) * output_tile_columns + final_x_begin / kTileSide + tile_x;
            noc.async_write(
                CoreLocalMem<uint32_t>(plane_addr + source_tile * kTileBytes),
                output,
                kTileBytes,
                {},
                {.page_id = output_tile_base + destination_tile});
            if (++outstanding == kWriteBatchTiles) {
                noc.async_write_barrier();
                outstanding = 0;
            }
        }
    }
    if (outstanding != 0) {
        noc.async_write_barrier();
    }
    return true;
}

template <typename OutputAccessor>
ALWI void write_band(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t output_tile_base,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& source,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    const uint32_t noc_scratch_addr) {
    if (write_band_full_tiles(
            output_args,
            output_addr,
            output_tile_columns,
            output_tile_base,
            plane_addr,
            plane_tile_columns,
            source,
            final_y_begin,
            final_y_length,
            final_x_begin,
            final_x_length)) {
        return;
    }
    write_band_fragmented(
        output_args,
        output_addr,
        output_tile_columns,
        output_tile_base,
        plane_addr,
        plane_tile_columns,
        source,
        final_y_begin,
        final_y_length,
        final_x_begin,
        final_x_length,
        noc_scratch_addr);
}

#ifdef ILWT_2D
[[nodiscard]] ALWI float read_plane_value(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const uint32_t y,
    const uint32_t x) {
    ASSERT(y >= stored.y_begin && y < stored.y_begin + stored.y_length);
    ASSERT(x >= stored.x_begin && x < stored.x_begin + stored.x_length);
    const uint32_t local_y = y - aligned_begin(stored.y_begin);
    const uint32_t local_x = x - aligned_begin(stored.x_begin);
    const auto* plane = reinterpret_cast<const volatile tt_l1_ptr float*>(plane_addr);
    return plane[tiled_element_offset(local_y, local_x, plane_tile_columns)];
}

struct TiledRowCursor {
    const volatile tt_l1_ptr float* plane;
    uint32_t tile_columns;
    uint32_t local_y;
    uint32_t local_x;
    uint32_t physical;
};

[[nodiscard]] ALWI TiledRowCursor make_tiled_row_cursor(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const uint32_t y,
    const uint32_t x) {
    ASSERT(y >= stored.y_begin && y < stored.y_begin + stored.y_length);
    ASSERT(x >= stored.x_begin && x < stored.x_begin + stored.x_length);
    const uint32_t local_y = y - aligned_begin(stored.y_begin);
    const uint32_t local_x = x - aligned_begin(stored.x_begin);
    return TiledRowCursor{
        .plane = reinterpret_cast<const volatile tt_l1_ptr float*>(plane_addr),
        .tile_columns = plane_tile_columns,
        .local_y = local_y,
        .local_x = local_x,
        .physical = tiled_element_offset(local_y, local_x, plane_tile_columns),
    };
}

[[nodiscard]] ALWI float read_and_advance(TiledRowCursor& cursor) {
    const float value = cursor.plane[cursor.physical];
    ++cursor.local_x;
    if (cursor.local_x % kFaceSide == 0) {
        cursor.physical = tiled_element_offset(cursor.local_y, cursor.local_x, cursor.tile_columns);
    } else {
        ++cursor.physical;
    }
    return value;
}

ALWI void fill_complete_interleaved_tile(
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t* parity_slots,
    const Rect* parity_sources,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const uint32_t tile_addr) {
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(tile_addr);
    const uint32_t padded_x = tile_x + pad_x;
    const uint32_t first_parity_x = padded_x & 1U;
    const uint32_t first_polyphase_x = padded_x / 2;
    const uint32_t second_polyphase_x = (padded_x + 1) / 2;

    for (uint32_t local_y = 0; local_y < kTileSide; ++local_y) {
        const uint32_t padded_y = tile_y + local_y + pad_y;
        const uint32_t parity_y = padded_y & 1U;
        const uint32_t polyphase_y = padded_y / 2;
        const uint32_t first_parity = 2 * parity_y + first_parity_x;
        const uint32_t second_parity = 2 * parity_y + (first_parity_x ^ 1U);
        const uint32_t first_slot = parity_slots[first_parity];
        const uint32_t second_slot = parity_slots[second_parity];
        TiledRowCursor first = make_tiled_row_cursor(
            plane_addrs[first_slot],
            plane_tile_columns[first_slot],
            parity_sources[first_parity],
            polyphase_y,
            first_polyphase_x);
        TiledRowCursor second = make_tiled_row_cursor(
            plane_addrs[second_slot],
            plane_tile_columns[second_slot],
            parity_sources[second_parity],
            polyphase_y,
            second_polyphase_x);
        auto* left_face_row = tile + tile_element_offset(local_y, 0);
        auto* right_face_row = tile + tile_element_offset(local_y, kFaceSide);
#pragma GCC unroll 8
        for (uint32_t pair = 0; pair < kFaceSide / 2; ++pair) {
            left_face_row[2 * pair] = read_and_advance(first);
            left_face_row[2 * pair + 1] = read_and_advance(second);
        }
#pragma GCC unroll 8
        for (uint32_t pair = 0; pair < kFaceSide / 2; ++pair) {
            right_face_row[2 * pair] = read_and_advance(first);
            right_face_row[2 * pair + 1] = read_and_advance(second);
        }
    }
}

template <typename OutputAccessor>
ALWI void write_interleaved_output(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t output_tile_base,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t* parity_slots,
    const Rect* parity_sources,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const uint32_t scratch_addr) {
    const auto output = TensorAccessor(output_args, output_addr, kTileBytes);
    Noc noc;
    const uint32_t tile_y_begin = aligned_begin(final_y_begin);
    const uint32_t tile_y_end = aligned_end(final_y_begin, final_y_length);
    const uint32_t tile_x_begin = aligned_begin(final_x_begin);
    const uint32_t tile_x_end = aligned_end(final_x_begin, final_x_length);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr);

    for (uint32_t tile_y = tile_y_begin; tile_y < tile_y_end; tile_y += kTileSide) {
        for (uint32_t tile_x = tile_x_begin; tile_x < tile_x_end; tile_x += kTileSide) {
            const uint32_t y_end = std::min(tile_y + kTileSide, final_y_begin + final_y_length);
            const uint32_t x_end = std::min(tile_x + kTileSide, final_x_begin + final_x_length);
            const uint32_t destination_tile = (tile_y / kTileSide) * output_tile_columns + tile_x / kTileSide;
            const bool complete = tile_y >= final_y_begin && tile_x >= final_x_begin &&
                                  tile_y + kTileSide <= final_y_begin + final_y_length &&
                                  tile_x + kTileSide <= final_x_begin + final_x_length;
            if (complete) {
                fill_complete_interleaved_tile(
                    plane_addrs,
                    plane_tile_columns,
                    parity_slots,
                    parity_sources,
                    tile_y,
                    tile_x,
                    pad_y,
                    pad_x,
                    scratch_addr);
                noc.async_write(
                    CoreLocalMem<uint32_t>(scratch_addr),
                    output,
                    kTileBytes,
                    {},
                    {.page_id = output_tile_base + destination_tile});
            } else {
                for (uint32_t y = std::max(tile_y, final_y_begin); y < y_end; ++y) {
                    const uint32_t padded_y = y + pad_y;
                    const uint32_t parity_y = padded_y & 1U;
                    const uint32_t polyphase_y = padded_y / 2;
                    for (uint32_t x = std::max(tile_x, final_x_begin); x < x_end; ++x) {
                        const uint32_t padded_x = x + pad_x;
                        const uint32_t parity_x = padded_x & 1U;
                        const uint32_t polyphase_x = padded_x / 2;
                        const uint32_t parity = 2 * parity_y + parity_x;
                        const uint32_t slot = parity_slots[parity];
                        tile[tile_element_offset(y - tile_y, x - tile_x)] = read_plane_value(
                            plane_addrs[slot],
                            plane_tile_columns[slot],
                            parity_sources[parity],
                            polyphase_y,
                            polyphase_x);
                    }
                }
                const uint32_t valid_y_begin = std::max(tile_y, final_y_begin);
                const uint32_t valid_x_begin = std::max(tile_x, final_x_begin);
                for (uint32_t y = valid_y_begin; y < y_end; ++y) {
                    for (uint32_t x = valid_x_begin; x < x_end;) {
                        const uint32_t local_x = x - tile_x;
                        const uint32_t count = std::min(x_end - x, kFaceSide - local_x % kFaceSide);
                        const uint32_t byte_offset = tile_element_offset(y - tile_y, local_x) * sizeof(float);
                        noc.async_write(
                            CoreLocalMem<uint32_t>(scratch_addr + byte_offset),
                            output,
                            count * sizeof(float),
                            {},
                            {.page_id = output_tile_base + destination_tile, .offset_bytes = byte_offset});
                        x += count;
                    }
                }
            }
            noc.async_write_barrier();
        }
    }
}
#endif

}  // namespace

void kernel_main() {
    uint32_t plane_addrs[ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount];
    uint32_t plane_tile_columns[ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount];
    const uint32_t workspace_base =
        CircularBuffer(ttnn::operations::wavelet::device_protocol::kLwt2DWorkspaceCb).get_write_ptr();
    for (uint32_t slot = 0; slot < ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount; ++slot) {
        plane_addrs[slot] = workspace_base + get_arg_val<uint32_t>(slot);
        plane_tile_columns[slot] =
            get_arg_val<uint32_t>(ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount + slot);
    }
    constexpr uint32_t plane_arg_count = 2 * ttnn::operations::wavelet::device_protocol::kLwt2DPlaneCount;
    const uint32_t route_config_addr = get_arg_val<uint32_t>(plane_arg_count);
    const uint32_t band_config_addr = get_arg_val<uint32_t>(plane_arg_count + 1);
#ifdef ILWT_2D
    const uint32_t output_addr = get_arg_val<uint32_t>(plane_arg_count + 2);
    const uint32_t output_tile_columns = get_arg_val<uint32_t>(plane_arg_count + 3);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(plane_arg_count + 4);
    const uint32_t chunk_count = get_arg_val<uint32_t>(plane_arg_count + 5);
    const uint32_t route_count = get_arg_val<uint32_t>(plane_arg_count + 6);
    const uint32_t pad_y = get_arg_val<uint32_t>(plane_arg_count + 7);
    const uint32_t pad_x = get_arg_val<uint32_t>(plane_arg_count + 8);
    const uint32_t chunks_per_sample = get_arg_val<uint32_t>(plane_arg_count + 9);
    const uint32_t output_tiles_per_sample = get_arg_val<uint32_t>(plane_arg_count + 10);
#else
    uint32_t output_addrs[ttnn::operations::wavelet::device_protocol::kLwt2DBandCount];
    for (uint32_t band = 0; band < ttnn::operations::wavelet::device_protocol::kLwt2DBandCount; ++band) {
        output_addrs[band] = get_arg_val<uint32_t>(plane_arg_count + 2 + band);
    }
    const uint32_t output_tile_columns = get_arg_val<uint32_t>(plane_arg_count + 6);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(plane_arg_count + 7);
    const uint32_t chunk_count = get_arg_val<uint32_t>(plane_arg_count + 8);
    const uint32_t route_count = get_arg_val<uint32_t>(plane_arg_count + 9);
    const uint32_t chunks_per_sample = get_arg_val<uint32_t>(plane_arg_count + 10);
    const uint32_t output_tiles_per_sample = get_arg_val<uint32_t>(plane_arg_count + 11);
#endif

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(1);
    constexpr uint32_t cb_band_config = get_compile_time_arg_val(2);
    constexpr uint32_t cb_noc_scratch = get_compile_time_arg_val(3);
    constexpr auto route_args = TensorAccessorArgs<4>();
    constexpr auto band_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<band_args.next_compile_time_args_offset()>();
    constexpr uint32_t split_scratch_bytes = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    CircularBuffer output_buffer(cb_output);
    CircularBuffer sync_buffer(cb_sync);
    const uint32_t noc_scratch_addr = CircularBuffer(cb_noc_scratch).get_write_ptr();
    const uint32_t writer_config_addr = noc_scratch_addr + split_scratch_bytes / 2;

    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        const uint32_t global_work_item = chunk_begin + local_chunk;
        const uint32_t batch_index = global_work_item / chunks_per_sample;
        const uint32_t global_chunk = global_work_item - batch_index * chunks_per_sample;
        const uint32_t output_tile_base = batch_index * output_tiles_per_sample;
        output_buffer.wait_front(1);
        preload_config_pages(
            route_args,
            route_config_addr,
            ttnn::operations::wavelet::device_protocol::kLwt2DRouteConfigPageBytes,
            global_chunk * route_count,
            route_count,
            writer_config_addr);
        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
            const auto* route_words = reinterpret_cast<const uint32_t*>(
                writer_config_addr +
                route_index * ttnn::operations::wavelet::device_protocol::kLwt2DRouteConfigPageBytes);
            const ConfigWords route_config{route_words};
            const uint32_t flags = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteFlags];
            if ((flags & ttnn::operations::wavelet::device_protocol::kLwt2DRouteFlagMetadataOnly) != 0) {
                continue;
            }
            const uint32_t output_slot = route_words[ttnn::operations::wavelet::device_protocol::kLwt2DRouteOutputSlot];
            const Rect output{route_config, ttnn::operations::wavelet::device_protocol::kLwt2DRouteOutputRect};
            write_local_output(cb_output, plane_addrs[output_slot], plane_tile_columns[output_slot], output);
            sync_buffer.reserve_back(1);
            sync_buffer.push_back(1);
        }

        uint32_t band_words[ttnn::operations::wavelet::device_protocol::kLwt2DBandConfigWordCount];
        load_config_page(
            band_args,
            band_config_addr,
            ttnn::operations::wavelet::device_protocol::kLwt2DBandConfigPageBytes,
            global_chunk,
            cb_band_config,
            band_words,
            ttnn::operations::wavelet::device_protocol::kLwt2DBandConfigWordCount);
        const uint32_t final_y_begin = band_words[ttnn::operations::wavelet::device_protocol::kLwt2DBandFinalYBegin];
        const uint32_t final_y_length = band_words[ttnn::operations::wavelet::device_protocol::kLwt2DBandFinalYLength];
        const uint32_t final_x_begin = band_words[ttnn::operations::wavelet::device_protocol::kLwt2DBandFinalXBegin];
        const uint32_t final_x_length = band_words[ttnn::operations::wavelet::device_protocol::kLwt2DBandFinalXLength];
        constexpr uint32_t band_offsets[ttnn::operations::wavelet::device_protocol::kLwt2DBandCount] = {
            ttnn::operations::wavelet::device_protocol::kLwt2DBandLl,
            ttnn::operations::wavelet::device_protocol::kLwt2DBandLh,
            ttnn::operations::wavelet::device_protocol::kLwt2DBandHl,
            ttnn::operations::wavelet::device_protocol::kLwt2DBandHh,
        };
        const ConfigWords band_config{band_words};
#ifdef ILWT_2D
        uint32_t parity_slots[ttnn::operations::wavelet::device_protocol::kLwt2DBandCount];
        Rect parity_sources[ttnn::operations::wavelet::device_protocol::kLwt2DBandCount];
        for (uint32_t parity = 0; parity < ttnn::operations::wavelet::device_protocol::kLwt2DBandCount; ++parity) {
            const uint32_t band_offset = band_offsets[parity];
            parity_slots[parity] =
                band_words[band_offset + ttnn::operations::wavelet::device_protocol::kLwt2DBandSourceSlot];
            parity_sources[parity] =
                Rect{band_config, band_offset + ttnn::operations::wavelet::device_protocol::kLwt2DBandSourceRect};
        }
        write_interleaved_output(
            output_args,
            output_addr,
            output_tile_columns,
            output_tile_base,
            plane_addrs,
            plane_tile_columns,
            parity_slots,
            parity_sources,
            final_y_begin,
            final_y_length,
            final_x_begin,
            final_x_length,
            pad_y,
            pad_x,
            noc_scratch_addr);
#else
        for (uint32_t band = 0; band < ttnn::operations::wavelet::device_protocol::kLwt2DBandCount; ++band) {
            const uint32_t band_offset = band_offsets[band];
            const uint32_t source_slot =
                band_words[band_offset + ttnn::operations::wavelet::device_protocol::kLwt2DBandSourceSlot];
            const Rect source{
                band_config, band_offset + ttnn::operations::wavelet::device_protocol::kLwt2DBandSourceRect};
            write_band(
                output_args,
                output_addrs[band],
                output_tile_columns,
                output_tile_base,
                plane_addrs[source_slot],
                plane_tile_columns[source_slot],
                source,
                final_y_begin,
                final_y_length,
                final_x_begin,
                final_x_length,
                noc_scratch_addr);
        }
#endif
        // Release the reader only after every final band has stopped reading
        // this chunk's workspace.
        sync_buffer.reserve_back(1);
        sync_buffer.push_back(1);
    }
}
