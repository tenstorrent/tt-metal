// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kFace = 16;
constexpr uint32_t kFaceElements = kFace * kFace;
constexpr uint32_t kFp32TileBytes = kTileHeight * kTileWidth * sizeof(uint32_t);
constexpr uint32_t kBf16TileBytes = kTileHeight * kTileWidth * sizeof(uint16_t);
constexpr uint16_t kBf16One = 0x3f80;
constexpr uint16_t kBf16Zero = 0;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
    return ((row / kFace) * 2 + col / kFace) * kFaceElements + (row % kFace) * kFace + col % kFace;
}

uint16_t fp32_bits_to_bf16(uint32_t bits) {
    // Round-to-nearest-even, matching the hardware FP32 -> BF16 pack conversion.
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

struct Geometry {
    uint32_t video_n_real;
    uint32_t frames;
    uint32_t height;
    uint32_t width;
};

struct Lookup {
    bool identity;
    uint32_t page;
    uint32_t element;
};

Lookup self_lookup(
    uint32_t global_token,
    uint32_t global_channel,
    const Geometry& g,
    uint32_t position_tiles,
    uint32_t rate_tiles) {
    if (global_token >= g.video_n_real || global_channel < 4 || g.height == 0 || g.width == 0) {
        return {true, 0, 0};
    }
    const uint32_t hw = g.height * g.width;
    const uint32_t t = global_token / hw;
    const uint32_t spatial = global_token % hw;
    const uint32_t h = spatial / g.width;
    const uint32_t w = spatial % g.width;
    if (t >= g.frames) {
        return {true, 0, 0};
    }

    // precompute_freqs_cis transposes [axis, rate] to [rate, axis] before
    // flattening, then repeat_interleave(2). Four identity channels precede
    // the 3 * 682 unique self-RoPE frequencies.
    const uint32_t unique = (global_channel - 4) / 2;
    const uint32_t axis = unique % 3;
    const uint32_t rate = unique / 3;
    const uint32_t position = axis == 0 ? t : (axis == 1 ? h : w);
    if (position >= position_tiles * kTileHeight || rate >= rate_tiles * kTileWidth) {
        return {true, 0, 0};
    }
    const uint32_t page =
        axis * position_tiles * rate_tiles + (position / kTileHeight) * rate_tiles + rate / kTileWidth;
    return {false, page, tile_element_index(position % kTileHeight, rate % kTileWidth)};
}

Lookup cross_lookup(
    uint32_t global_token,
    uint32_t global_channel,
    const Geometry& g,
    uint32_t position_tiles,
    uint32_t rate_tiles) {
    if (global_token >= g.video_n_real || g.height == 0 || g.width == 0) {
        return {true, 0, 0};
    }
    const uint32_t hw = g.height * g.width;
    const uint32_t t = global_token / hw;
    if (t >= g.frames) {
        return {true, 0, 0};
    }

    // Cross PE has one temporal axis and 1024 rates; each rate is duplicated
    // into its adjacent interleaved rotary pair.
    const uint32_t rate = global_channel / 2;
    if (t >= position_tiles * kTileHeight || rate >= rate_tiles * kTileWidth) {
        return {true, 0, 0};
    }
    const uint32_t page = (t / kTileHeight) * rate_tiles + rate / kTileWidth;
    return {false, page, tile_element_index(t % kTileHeight, rate % kTileWidth)};
}

template <typename SourceAccessor, typename OutputAccessor>
void materialize_tile(
    Noc& noc,
    const SourceAccessor& source,
    const OutputAccessor& destination,
    CircularBuffer& output_cb,
    CircularBuffer& cache_cb,
    uint32_t output_tile,
    bool is_self,
    bool is_sin,
    const Geometry& geometry,
    uint32_t position_tiles,
    uint32_t rate_tiles,
    uint32_t heads,
    uint32_t seq_tiles,
    uint32_t dim_tiles,
    uint32_t sp_coord,
    uint32_t tp_coord,
    uint32_t scratch_entries) {
    const uint32_t head = output_tile / (seq_tiles * dim_tiles);
    const uint32_t rem = output_tile % (seq_tiles * dim_tiles);
    const uint32_t seq_tile = rem / dim_tiles;
    const uint32_t dim_tile = rem % dim_tiles;

    uint32_t pages[16];
    uint32_t page_count = 0;
    for (uint32_t row = 0; row < kTileHeight; ++row) {
        const uint32_t global_token = sp_coord * seq_tiles * kTileHeight + seq_tile * kTileHeight + row;
        for (uint32_t col = 0; col < kTileWidth; ++col) {
            const uint32_t global_head = tp_coord * heads + head;
            const uint32_t head_dim = dim_tiles * kTileWidth;
            const uint32_t global_channel = global_head * head_dim + dim_tile * kTileWidth + col;
            const Lookup lookup = is_self
                                      ? self_lookup(
                                            global_token, global_channel, geometry, position_tiles, rate_tiles)
                                      : cross_lookup(
                                            global_token, global_channel, geometry, position_tiles, rate_tiles);
            if (lookup.identity) {
                continue;
            }
            bool found = false;
            for (uint32_t i = 0; i < page_count; ++i) {
                found |= pages[i] == lookup.page;
            }
            if (!found && page_count < scratch_entries) {
                pages[page_count++] = lookup.page;
            }
        }
    }

    cache_cb.reserve_back(scratch_entries);
    const uint32_t cache_base = cache_cb.get_write_ptr();
    constexpr uint32_t cache_tile_bytes = kFp32TileBytes;
    for (uint32_t i = 0; i < page_count; ++i) {
        noc.async_read(
            source,
            CoreLocalMem<uint32_t>(cache_base + i * cache_tile_bytes),
            cache_tile_bytes,
            {.page_id = pages[i]},
            {});
    }
    noc.async_read_barrier();
    cache_cb.push_back(scratch_entries);
    cache_cb.wait_front(scratch_entries);
    invalidate_l1_cache();

    output_cb.reserve_back(1);
    const uint32_t output_write = output_cb.get_write_ptr();
    CoreLocalMem<volatile uint16_t> output_data(output_write);
    for (uint32_t row = 0; row < kTileHeight; ++row) {
        const uint32_t global_token = sp_coord * seq_tiles * kTileHeight + seq_tile * kTileHeight + row;
        for (uint32_t col = 0; col < kTileWidth; ++col) {
            const uint32_t global_head = tp_coord * heads + head;
            const uint32_t head_dim = dim_tiles * kTileWidth;
            const uint32_t global_channel = global_head * head_dim + dim_tile * kTileWidth + col;
            const Lookup lookup = is_self
                                      ? self_lookup(
                                            global_token, global_channel, geometry, position_tiles, rate_tiles)
                                      : cross_lookup(
                                            global_token, global_channel, geometry, position_tiles, rate_tiles);
            uint16_t value = is_sin ? kBf16Zero : kBf16One;
            if (!lookup.identity) {
                uint32_t cache_slot = 0;
                for (; cache_slot < page_count; ++cache_slot) {
                    if (pages[cache_slot] == lookup.page) {
                        break;
                    }
                }
                if (cache_slot < page_count) {
                    CoreLocalMem<volatile uint32_t> tile(cache_base + cache_slot * cache_tile_bytes);
                    value = fp32_bits_to_bf16(tile[lookup.element]);
                }
            }
            output_data[tile_element_index(row, col)] = value;
        }
    }

    output_cb.push_back(1);
    output_cb.wait_front(1);
    noc.async_write(
        CoreLocalMem<uint32_t>(output_cb.get_read_ptr()),
        destination,
        kBf16TileBytes,
        {},
        {.page_id = output_tile});
    noc.async_write_barrier();
    output_cb.pop_front(1);
    cache_cb.pop_front(scratch_entries);
}

}  // namespace

void kernel_main() {
    Noc noc;

    const uint32_t start_tile = get_arg(args::start_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t self_position_t = get_arg(args::self_position_t);
    constexpr uint32_t self_rate_t = get_arg(args::self_rate_t);
    constexpr uint32_t cross_position_t = get_arg(args::cross_position_t);
    constexpr uint32_t cross_rate_t = get_arg(args::cross_rate_t);
    constexpr uint32_t self_heads = get_arg(args::self_heads);
    constexpr uint32_t self_seq_t = get_arg(args::self_seq_t);
    constexpr uint32_t self_dim_t = get_arg(args::self_dim_t);
    constexpr uint32_t cross_heads = get_arg(args::cross_heads);
    constexpr uint32_t cross_seq_t = get_arg(args::cross_seq_t);
    constexpr uint32_t cross_dim_t = get_arg(args::cross_dim_t);
    constexpr uint32_t self_tiles = get_arg(args::self_tiles);
    constexpr uint32_t cross_tiles = get_arg(args::cross_tiles);
    constexpr uint32_t sp_coord = get_arg(args::sp_coord);
    constexpr uint32_t tp_coord = get_arg(args::tp_coord);
    constexpr uint32_t scratch_entries = get_arg(args::scratch_entries);

    const auto self_cos_input = TensorAccessor(tensor::self_cos_input);
    const auto self_sin_input = TensorAccessor(tensor::self_sin_input);
    const auto cross_cos_input = TensorAccessor(tensor::cross_cos_input);
    const auto cross_sin_input = TensorAccessor(tensor::cross_sin_input);
    const auto metadata = TensorAccessor(tensor::metadata);
    const auto self_cos_output = TensorAccessor(tensor::self_cos_output);
    const auto self_sin_output = TensorAccessor(tensor::self_sin_output);
    const auto cross_cos_output = TensorAccessor(tensor::cross_cos_output);
    const auto cross_sin_output = TensorAccessor(tensor::cross_sin_output);

    CircularBuffer output_cb(dfb::output);
    CircularBuffer cache_cb(dfb::cache);
    CircularBuffer meta_cb(dfb::meta);

    meta_cb.reserve_back(1);
    const uint32_t meta_ptr = meta_cb.get_write_ptr();
    noc.async_read(metadata, CoreLocalMem<uint32_t>(meta_ptr), 16, {.page_id = 0}, {});
    noc.async_read_barrier();
    meta_cb.push_back(1);
    meta_cb.wait_front(1);
    invalidate_l1_cache();
    CoreLocalMem<volatile uint32_t> meta(meta_ptr);
    const Geometry geometry{meta[0], meta[1], meta[2], meta[3]};

    const uint32_t self_cos_end = self_tiles;
    const uint32_t self_sin_end = 2 * self_tiles;
    const uint32_t cross_cos_end = self_sin_end + cross_tiles;
    const uint32_t end_tile = start_tile + num_tiles;
    for (uint32_t work = start_tile; work < end_tile; ++work) {
        if (work < self_cos_end) {
            materialize_tile(
                noc,
                self_cos_input,
                self_cos_output,
                output_cb,
                cache_cb,
                work,
                true,
                false,
                geometry,
                self_position_t,
                self_rate_t,
                self_heads,
                self_seq_t,
                self_dim_t,
                sp_coord,
                tp_coord,
                scratch_entries);
        } else if (work < self_sin_end) {
            materialize_tile(
                noc,
                self_sin_input,
                self_sin_output,
                output_cb,
                cache_cb,
                work - self_cos_end,
                true,
                true,
                geometry,
                self_position_t,
                self_rate_t,
                self_heads,
                self_seq_t,
                self_dim_t,
                sp_coord,
                tp_coord,
                scratch_entries);
        } else if (work < cross_cos_end) {
            materialize_tile(
                noc,
                cross_cos_input,
                cross_cos_output,
                output_cb,
                cache_cb,
                work - self_sin_end,
                false,
                false,
                geometry,
                cross_position_t,
                cross_rate_t,
                cross_heads,
                cross_seq_t,
                cross_dim_t,
                sp_coord,
                tp_coord,
                scratch_entries);
        } else {
            materialize_tile(
                noc,
                cross_sin_input,
                cross_sin_output,
                output_cb,
                cache_cb,
                work - cross_cos_end,
                false,
                true,
                geometry,
                cross_position_t,
                cross_rate_t,
                cross_heads,
                cross_seq_t,
                cross_dim_t,
                sp_coord,
                tp_coord,
                scratch_entries);
        }
    }

    meta_cb.pop_front(1);
}
