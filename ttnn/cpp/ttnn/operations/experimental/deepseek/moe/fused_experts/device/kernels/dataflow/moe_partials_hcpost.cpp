// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/tensor_accessor.h"

// persistent_moe_request with the hyper-connection post step fused, on one stream-collapse core: this core's
// part of the row is its columns of the width-sharded streams, and its new streams are
//     new[h] = post[h] * x + sum_g comb[g, h] * old[g],     x = the sum of the op's chips' partials of the part
// (compute/moe_partials_hcpost.cpp, which views each part as one 32x32 tile through part_bytes pages). This kernel
//   1. reads post and comb (one 32x32 bf16 tile each; post[h] is element (h, 0), comb[g, h] element (g, h), all
//      in face 0) and stages each value as element 0 of its own scalar page -- the compute broadcasts that
//      element: cb_post page h, cb_comb page g * num_streams + h;
//   2. copies this core's shard of the old streams (num_streams rows of part_bytes) into cb_s;
//   3. once every chip's flag is set, publishes the partials slots (cb_slots, over the partials shard); a block
//      (rows > 1) publishes each slot as its flag is set, in chip order;
//   4. writes the new streams (cb_out, row h at h * part_bytes) into this core's shard of the output, and clears
//      the flags: the next request, which is the next time the op sends here, comes after this op has finished.
//
// Compile-time args:
//   0: num_slots (the op's chips)   1: tiles (per row of a part: 1)   2: num_streams   3: part_bytes
//   4: cb_slots   5: cb_s   6: cb_post   7: cb_comb   8: cb_out   9: cb_tmp (post's and comb's face-0 heads)
//   10: comb scalar page bytes   11: post scalar page bytes   12: rows   13: part pages (rows > 1)
//   14+: TensorAccessorArgs(post), then TensorAccessorArgs(comb)
//
// A several-row block (rows > 1): post and comb are [1, rows, ...], one tile per row j (page j), and each value
// is staged as column 0 of its own 32x32 tile (rows R in face 0, R + 16 in face 2) holding row R % rows's value --
// the compute's column broadcast (see compute/moe_partials_hcpost.cpp). Those tiles are scalar_tile_bytes apart, an
// odd multiple of 16 B past 1 KB, so a tile's face 2 column 0 and the next tile's face 0 column 0 interleave. Each
// row of a part (cb_s, cb_slots, cb_out) is part_pages pages of part_bytes / part_pages.
//
// Runtime args:
//   0: skip (a stop request, which gets no partials)   1: partials buffer address   2: post address
//   3: comb address   4: old streams address   5: new streams address (both this core's shard)

namespace {
constexpr uint32_t kFaceHeadBytes = 128;  // face 0's first 4 rows of 16 bf16: every (r < 4, c < 16) element

FORCE_INLINE void put_scalar(uint32_t l1, uint16_t value) {
    *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1) = value;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t streams = get_compile_time_arg_val(2);
    constexpr uint32_t part_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(4);
    constexpr uint32_t cb_s = get_compile_time_arg_val(5);
    constexpr uint32_t cb_post = get_compile_time_arg_val(6);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);
    constexpr uint32_t scalar_tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t post_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t rows = get_compile_time_arg_val(12);
    constexpr uint32_t part_pages = get_compile_time_arg_val(13);
    constexpr auto post_args = TensorAccessorArgs<14>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t row_tiles = streams * tiles;
    static_assert(streams <= 4, "post and comb are read from face 0's first 4 rows");
    // A row of the streams' (and output's) part in CB pages, and the slots'.
    constexpr uint32_t stream_pages = rows > 1 ? streams * part_pages : row_tiles;
    constexpr uint32_t slot_pages = rows > 1 ? num_slots * part_pages : num_slots * tiles;

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    const uint32_t partials_addr = get_arg_val<uint32_t>(1);
    const auto post = TensorAccessor(post_args, get_arg_val<uint32_t>(2));
    const auto comb = TensorAccessor(comb_args, get_arg_val<uint32_t>(3));
    const uint32_t old_addr = get_arg_val<uint32_t>(4);
    const uint32_t new_addr = get_arg_val<uint32_t>(5);
    auto* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(partials_addr + num_slots * part_bytes);

    // 1. and 2.
    const uint32_t tmp = get_write_ptr(cb_tmp);
    if constexpr (rows > 1) {
        // Row j's post head at j, its comb head at rows + j.
        for (uint32_t j = 0; j < rows; ++j) {
            noc_async_read(post.get_noc_addr(j), tmp + j * kFaceHeadBytes, kFaceHeadBytes);
            noc_async_read(comb.get_noc_addr(j), tmp + (rows + j) * kFaceHeadBytes, kFaceHeadBytes);
        }
    } else {
        noc_async_read(post.get_noc_addr(0), tmp, kFaceHeadBytes);
        noc_async_read(comb.get_noc_addr(0), tmp + kFaceHeadBytes, kFaceHeadBytes);
    }
    cb_reserve_back(cb_s, stream_pages);
    noc_async_read(get_noc_addr(old_addr), get_write_ptr(cb_s), streams * part_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_s, stream_pages);

    auto* head = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
    if constexpr (rows > 1) {
        constexpr uint32_t kHeadElems = kFaceHeadBytes / 2u;
        constexpr uint32_t kFace2 = 1024;  // face 2's offset in a 32x32 bf16 tile
        // Column 0 of the tile at l1: row R (and R + 16) gets row R % rows's value at element `at` of its head.
        auto put_column = [&](uint32_t l1, uint32_t first_head, uint32_t at) {
            for (uint32_t r = 0; r < 16; ++r) {
                const uint16_t value = head[(first_head + r % rows) * kHeadElems + at];
                put_scalar(l1 + r * 32u, value);
                put_scalar(l1 + kFace2 + r * 32u, value);
            }
        };
        cb_reserve_back(cb_comb, streams * streams);
        const uint32_t comb_l1 = get_write_ptr(cb_comb);
        for (uint32_t g = 0; g < streams; ++g) {
            for (uint32_t h = 0; h < streams; ++h) {
                put_column(comb_l1 + (g * streams + h) * scalar_tile_bytes, rows, g * 16u + h);
            }
        }
        cb_push_back(cb_comb, streams * streams);
        cb_reserve_back(cb_post, streams);
        const uint32_t post_l1 = get_write_ptr(cb_post);
        for (uint32_t h = 0; h < streams; ++h) {
            put_column(post_l1 + h * post_tile_bytes, 0, h * 16u);
        }
        cb_push_back(cb_post, streams);
    } else {
        cb_reserve_back(cb_comb, streams * streams);
        const uint32_t comb_l1 = get_write_ptr(cb_comb);
        for (uint32_t g = 0; g < streams; ++g) {
            for (uint32_t h = 0; h < streams; ++h) {
                put_scalar(comb_l1 + (g * streams + h) * scalar_tile_bytes, head[kFaceHeadBytes / 2u + g * 16u + h]);
            }
        }
        cb_push_back(cb_comb, streams * streams);
        cb_reserve_back(cb_post, streams);
        const uint32_t post_l1 = get_write_ptr(cb_post);
        for (uint32_t h = 0; h < streams; ++h) {
            put_scalar(post_l1 + h * post_tile_bytes, head[h * 16u]);
        }
        cb_push_back(cb_post, streams);
    }

    // 3.
    {
        DeviceZoneScopedN("MOE_SUM_WAIT");
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            do {
                invalidate_l1_cache();
            } while (flags[slot] == 0u);
            if constexpr (rows > 1) {
                // A block's slots one at a time, for the compute to sum as they land.
                cb_push_back(cb_slots, part_pages);
            }
        }
    }
    if constexpr (rows == 1) {
        cb_push_back(cb_slots, slot_pages);
    }

    // 4.
    {
        DeviceZoneScopedN("MOE_SUM_WRITE");
        cb_wait_front(cb_out, stream_pages);
        noc_async_write(get_read_ptr(cb_out), get_noc_addr(new_addr), streams * part_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, stream_pages);
    }
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        flags[slot] = 0u;
    }
}
