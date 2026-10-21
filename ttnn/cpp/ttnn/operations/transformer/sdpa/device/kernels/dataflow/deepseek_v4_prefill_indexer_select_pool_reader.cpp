// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_indexer_select pool-mode reader. Per query: turns its candidate blocks into its pool list
// (handed to the writer, which maps selected pool positions back to entries) and the pool size (to compute), builds
// its q as [heads, dims] tiles and its gates as row 0 of a tile, then gathers the idx_keys rows of its pool. The
// cost follows the pool, not the number of entries the query can see.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"
#include "metadata_scalar_read.hpp"

namespace dis = deepseek_v4_prefill_indexer_select;

void kernel_main() {
    using namespace tt::constants;
    constexpr uint32_t num_heads = get_compile_time_arg_val(dis::pool_reader_ct_arg::NUM_HEADS);
    constexpr uint32_t head_dim = get_compile_time_arg_val(dis::pool_reader_ct_arg::HEAD_DIM);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(dis::pool_reader_ct_arg::NUM_TOKENS);
    constexpr uint32_t rate = get_compile_time_arg_val(dis::pool_reader_ct_arg::RATE);
    constexpr uint32_t cb_q = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_Q);
    constexpr uint32_t cb_w = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_W);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_K_RM);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_NEGINF);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_META);
    constexpr uint32_t cb_start = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_START);
    constexpr uint32_t cb_list = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_LIST);
    constexpr uint32_t cb_bounce = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_BOUNCE);
    constexpr uint32_t cb_cands = get_compile_time_arg_val(dis::pool_reader_ct_arg::CB_CANDS);
    static_assert(num_heads == TILE_HEIGHT, "pool mode holds every head in one tile row");

    constexpr auto q_args = TensorAccessorArgs<dis::pool_reader_ct_arg::END, dis::pool_reader_common_arg::END>();
    constexpr auto w_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto keys_args =
        TensorAccessorArgs<w_args.next_compile_time_args_offset(), w_args.next_common_runtime_args_offset()>();
    constexpr auto start_args =
        TensorAccessorArgs<keys_args.next_compile_time_args_offset(), keys_args.next_common_runtime_args_offset()>();
    constexpr auto cands_args =
        TensorAccessorArgs<start_args.next_compile_time_args_offset(), start_args.next_common_runtime_args_offset()>();

    const uint32_t start_addr = get_common_arg_val<uint32_t>(dis::pool_reader_common_arg::START_ADDRESS);
    const uint32_t num_entries = get_common_arg_val<uint32_t>(dis::pool_reader_common_arg::NUM_ENTRIES);
    const uint32_t first_query = get_arg_val<uint32_t>(dis::pool_core_arg::FIRST_QUERY);
    const uint32_t num_queries = get_arg_val<uint32_t>(dis::pool_core_arg::NUM_QUERIES);

    constexpr uint32_t tile_bytes = TILE_HW * sizeof(uint16_t);
    constexpr uint32_t face_bytes = FACE_HW * sizeof(uint16_t);
    constexpr uint32_t face_row_bytes = FACE_WIDTH * sizeof(uint16_t);
    constexpr uint32_t head_dim_tiles = head_dim / TILE_WIDTH;
    constexpr uint32_t token_tiles = num_tokens / TILE_HEIGHT;
    constexpr uint32_t key_row_bytes = head_dim * sizeof(uint16_t);
    constexpr uint32_t step_rows = dis::STEP_KEY_TILES * TILE_HEIGHT;
    constexpr uint32_t steps_per_block = dis::TOPK / step_rows;
    constexpr uint32_t cand_row_bytes = dis::CAND_TOPK * sizeof(uint32_t);
    // Bounce slot (head, dim tile, half) holds the aligned 64 B around one face row of a q tile; two more hold the
    // gate row.
    constexpr uint32_t q_pieces = num_heads * head_dim_tiles * 2;

    Noc noc;
    experimental::CB q_cb(cb_q), w_cb(cb_w), k_cb(cb_k_rm), neginf_cb(cb_neginf), meta_cb(cb_meta), start_cb(cb_start),
        list_cb(cb_list), bounce_cb(cb_bounce), cands_cb(cb_cands);
    const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(dis::pool_reader_common_arg::Q_ADDRESS));
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(dis::pool_reader_common_arg::W_ADDRESS));
    const auto keys =
        TensorAccessor(keys_args, get_common_arg_val<uint32_t>(dis::pool_reader_common_arg::KEYS_ADDRESS));
    const auto cands =
        TensorAccessor(cands_args, get_common_arg_val<uint32_t>(dis::pool_reader_common_arg::CANDS_ADDRESS));

    start_cb.reserve_back(1);
    const uint32_t start =
        trace_metadata::read_metadata_scalar_u32(noc, start_args, start_addr, start_cb.get_write_ptr());

    // The all -inf chunk the leader's emit seeds from when a query's pool is empty.
    neginf_cb.reserve_back(dis::NEGINF_TILES);
    {
        auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(neginf_cb.get_write_ptr());
        for (uint32_t i = 0; i < dis::NEGINF_TILES * tile_bytes / sizeof(uint32_t); ++i) {
            p[i] = 0xFF80FF80u;
        }
    }
    neginf_cb.push_back(dis::NEGINF_TILES);

    cands_cb.reserve_back(1);
    bounce_cb.reserve_back(1);
    const uint32_t cands_ptr = cands_cb.get_write_ptr();
    const uint32_t bounce_ptr = bounce_cb.get_write_ptr();

    for (uint32_t i = 0; i < num_queries; ++i) {
        const uint32_t token = first_query + i;
        const uint32_t q_tile = token / TILE_HEIGHT;
        const uint32_t r = token % TILE_HEIGHT;
        // The tile's face row holding row r (as 16-datum halves in faces 2 * (r / 16) and 2 * (r / 16) + 1), read
        // as the aligned 64 B pair of face rows around it.
        const uint32_t piece_offset = (r / FACE_HEIGHT) * 2 * face_bytes + ((r % FACE_HEIGHT) & ~1u) * face_row_bytes;
        const uint32_t piece_half = (r % 2) * face_row_bytes;

        noc.async_read(cands, CoreLocalMem<uint32_t>(cands_ptr), cand_row_bytes, {.page_id = token}, {});
        for (uint32_t h = 0; h < num_heads; ++h) {
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_read(
                        q,
                        CoreLocalMem<uint32_t>(
                            bounce_ptr + ((h * head_dim_tiles + d) * 2 + half) * dis::POOL_PIECE_BYTES),
                        dis::POOL_PIECE_BYTES,
                        {.page_id = (h * token_tiles + q_tile) * head_dim_tiles + d,
                         .offset_bytes = piece_offset + half * face_bytes},
                        {});
                }
            }
        }
        for (uint32_t half = 0; half < 2; ++half) {
            noc.async_read(
                w,
                CoreLocalMem<uint32_t>(bounce_ptr + (q_pieces + half) * dis::POOL_PIECE_BYTES),
                dis::POOL_PIECE_BYTES,
                {.page_id = q_tile, .offset_bytes = piece_offset + half * face_bytes},
                {});
        }
        noc.async_read_barrier();

        // Pool list: the visible candidate blocks, the newest (pinned by the candidate select, the only partial
        // one) last.
        const uint32_t seen = (start + token + 1) / rate;
        const uint32_t limit = seen < num_entries ? seen : num_entries;
        const uint32_t visible = (limit + dis::CAND_BLOCK - 1) / dis::CAND_BLOCK;
        list_cb.reserve_back(1);
        auto* list_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_cb.get_write_ptr());
        auto* list = list_page + dis::POOL_LIST_WORDS;
        auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cands_ptr);
        uint32_t n = 0;
        bool newest = false;
        for (uint32_t k = 0; k < dis::CAND_TOPK; ++k) {
            const uint32_t b = row[k];
            if (b >= visible) {
                continue;
            }
            if (b + 1 == visible) {
                newest = true;
            } else {
                list[n++] = b;
            }
        }
        uint32_t pool = n * dis::CAND_BLOCK;
        if (newest) {
            list[n++] = visible - 1;
            pool += limit - (visible - 1) * dis::CAND_BLOCK;
        }
        list_page[0] = n;
        list_cb.push_back(1);

        const uint32_t num_blocks = (pool + dis::TOPK - 1) / dis::TOPK;
        meta_cb.reserve_back(1);
        auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_cb.get_write_ptr());
        meta[dis::meta_word::NUM_BLOCKS] = num_blocks;
        meta[dis::meta_word::FIRST_ENTRY] = 0;
        meta[dis::meta_word::TILE_BLOCKS] = num_blocks;
        meta[dis::meta_word::LIMITS] = pool;
        meta_cb.push_back(1);

        // q as [heads, dims] tiles: row h of dim tile d is query row r of q tile (h, d).
        q_cb.reserve_back(head_dim_tiles);
        for (uint32_t d = 0; d < head_dim_tiles; ++d) {
            const uint32_t tile = q_cb.get_write_ptr() + d * tile_bytes;
            for (uint32_t h = 0; h < num_heads; ++h) {
                for (uint32_t half = 0; half < 2; ++half) {
                    auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        bounce_ptr + ((h * head_dim_tiles + d) * 2 + half) * dis::POOL_PIECE_BYTES + piece_half);
                    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        tile + ((h / FACE_HEIGHT) * 2 + half) * face_bytes + (h % FACE_HEIGHT) * face_row_bytes);
                    for (uint32_t j = 0; j < face_row_bytes / sizeof(uint32_t); ++j) {
                        dst[j] = src[j];
                    }
                }
            }
        }
        q_cb.push_back(head_dim_tiles);

        // Gates as row 0 of an otherwise zero tile (the reducing operand of the head sum).
        w_cb.reserve_back(1);
        {
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_cb.get_write_ptr());
            for (uint32_t j = 0; j < tile_bytes / sizeof(uint32_t); ++j) {
                dst[j] = 0;
            }
            for (uint32_t half = 0; half < 2; ++half) {
                auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    bounce_ptr + (q_pieces + half) * dis::POOL_PIECE_BYTES + piece_half);
                for (uint32_t j = 0; j < face_row_bytes / sizeof(uint32_t); ++j) {
                    dst[half * face_bytes / sizeof(uint32_t) + j] = src[j];
                }
            }
        }
        w_cb.push_back(1);

        // idx_keys rows of the pool, one step of step_rows positions at a time; positions past the pool are not
        // read, and their scores are never selected.
        for (uint32_t b = 0; b < num_blocks; ++b) {
            for (uint32_t step = 0; step < steps_per_block; ++step) {
                const uint32_t p0 = b * dis::TOPK + step * step_rows;
                k_cb.reserve_back(step_rows);
                const uint32_t k_ptr = k_cb.get_write_ptr();
                const uint32_t rows = p0 >= pool ? 0 : pool - p0 < step_rows ? pool - p0 : step_rows;
                for (uint32_t j = 0; j < rows; ++j) {
                    const uint32_t p = p0 + j;
                    noc.async_read(
                        keys,
                        CoreLocalMem<uint32_t>(k_ptr + j * key_row_bytes),
                        key_row_bytes,
                        {.page_id = list[p / dis::CAND_BLOCK] * dis::CAND_BLOCK + p % dis::CAND_BLOCK},
                        {});
                }
                noc.async_read_barrier();
                k_cb.push_back(step_rows);
            }
        }
    }
}
