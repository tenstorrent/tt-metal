// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"
#include "rope_table_rows.hpp"

// BLOCK_ROWS: see below; the ring is not touched.
//
// kv core `kv_index` owns column tile kv_index of the ring: it writes its normalized (and, do_rope,
// rotated) 1x32 tile into row cache_pos of kv_cache. The tile holding that row is read up front, while
// the matmul runs, and written back whole once the row is patched in.
void kernel_main() {
    constexpr uint32_t head_tiles = get_arg(args::head_tiles);
    constexpr uint32_t rope_tiles = get_arg(args::rope_tiles);
    constexpr uint32_t row_tile_bytes = get_arg(args::row_tile_bytes);
    constexpr uint32_t cache_tile_bytes = get_arg(args::cache_tile_bytes);
    constexpr uint32_t pos_page_bytes = get_arg(args::pos_page_bytes);
    const uint32_t kv_index = get_arg(args::kv_index);
    const bool do_rope = get_arg(args::do_rope) != 0;

    Noc noc;
    CircularBuffer gamma_cb(dfb::gamma);
    CircularBuffer cs_cb(dfb::cs);
    CircularBuffer cs_raw_cb(dfb::cs_raw);
    CircularBuffer out_cb(dfb::out);

#ifdef BLOCK_ROWS
    // A block of `rows` rows: gamma replicated over them, the ones for the per-row statistic, per-row RoPE rows,
    // and the kv rows written into rows [0, rows) of kv_cache's first tile row (the staging tensor; no position).
    constexpr uint32_t rows = get_arg(args::rows);
    CircularBuffer ones_cb(dfb::ones);
    const auto kv_gamma = TensorAccessor(tensor::kv_gamma);
    read_gamma_rows(kv_gamma, kv_index, rows, gamma_cb);
    if (do_rope) {
        const auto cos = TensorAccessor(tensor::cos);
        const auto sin = TensorAccessor(tensor::sin);
        read_rope_table_tiles(noc, cos, sin, kv_index + rope_tiles - head_tiles, 1u, cs_cb, row_tile_bytes);
    }
    noc.async_read_barrier();
    gamma_cb.push_back(1);
    fill_ones_tile(ones_cb);

    // Face f of the (rows x 32) tile is rows x 16 values, row-major: the same rows of face f of a 32x32 tile.
    constexpr uint32_t tile_face_bytes = 16 * 16 * 2;
    constexpr uint32_t block_face_bytes = rows * 16 * 2;
    const auto staging = TensorAccessor(tensor::kv_cache);
    out_cb.wait_front(1);
    const uint32_t block_addr = out_cb.get_read_ptr();
    for (uint32_t face = 0; face < 2; ++face) {
        noc_async_write(
            block_addr + face * block_face_bytes,
            staging.get_noc_addr(kv_index, face * tile_face_bytes),
            block_face_bytes);
    }
    noc.async_write_barrier();
    out_cb.pop_front(1);
    return;
#endif

    CircularBuffer cache_tile_cb(dfb::cache_tile);
    CircularBuffer pos_cb(dfb::pos);

    // gamma is resident in the tensor it borrows.
    gamma_cb.reserve_back(1);
    gamma_cb.push_back(1);

    const auto cache_pos = TensorAccessor(tensor::cache_pos);
    pos_cb.reserve_back(1);
    const uint32_t pos_addr = pos_cb.get_write_ptr();
    noc_async_read(cache_pos.get_noc_addr(0), pos_addr, pos_page_bytes);
    noc.async_read_barrier();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr)[0];
    pos_cb.push_back(1);
    pos_cb.wait_front(1);
    pos_cb.pop_front(1);

    constexpr uint32_t tile_rows = 32;
    const auto kv_cache = TensorAccessor(tensor::kv_cache);
    const uint64_t tile_noc_addr = kv_cache.get_noc_addr((pos / tile_rows) * head_tiles + kv_index);
    cache_tile_cb.reserve_back(1);
    const uint32_t tile_addr = cache_tile_cb.get_write_ptr();
    noc_async_read(tile_noc_addr, tile_addr, cache_tile_bytes);

    if (do_rope) {
        const auto cos = TensorAccessor(tensor::cos);
        const auto sin = TensorAccessor(tensor::sin);
        read_rope_table_rows(noc, cos, sin, kv_index + rope_tiles - head_tiles, 1u, cs_cb, cs_raw_cb, row_tile_bytes);
    }
    noc.async_read_barrier();

    // Row r of a 32x32 tile: its first 16 values are row r % 16 of face 2 * (r / 16), the rest the same
    // row of the face after it.
    constexpr uint32_t face_bytes = 16 * 16 * 2;
    constexpr uint32_t face_row_bytes = 16 * 2;
    const uint32_t r = pos % tile_rows;
    const uint32_t row_offset = (r / 16) * 2 * face_bytes + (r % 16) * face_row_bytes;
    out_cb.wait_front(1);
    const uint32_t row_addr = out_cb.get_read_ptr();
    copy_l1_words(tile_addr + row_offset, row_addr, face_row_bytes);
    copy_l1_words(tile_addr + row_offset + face_bytes, row_addr + face_row_bytes, face_row_bytes);
    out_cb.pop_front(1);

    noc_async_write(tile_addr, tile_noc_addr, cache_tile_bytes);
    noc.async_write_barrier();
    cache_tile_cb.push_back(1);
    cache_tile_cb.wait_front(1);
    cache_tile_cb.pop_front(1);
}
