// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_indexer_select reader. Per work unit (one query tile, one split of its visible blocks) it
// derives the causal limits from the device-side start position and hands them to compute, then streams the q
// heads, the per-head gates and the split's idx_keys rows. Only rows of visible blocks are read, so the cost
// follows the number of entries the chunk can see, not the buffer's capacity.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"
#include "metadata_scalar_read.hpp"

namespace dis = deepseek_v4_prefill_indexer_select;

void kernel_main() {
    using namespace tt::constants;
    constexpr uint32_t num_heads = get_compile_time_arg_val(dis::reader_ct_arg::NUM_HEADS);
    constexpr uint32_t head_dim = get_compile_time_arg_val(dis::reader_ct_arg::HEAD_DIM);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(dis::reader_ct_arg::NUM_TOKENS);
    constexpr uint32_t rate = get_compile_time_arg_val(dis::reader_ct_arg::RATE);
    constexpr uint32_t num_splits = get_compile_time_arg_val(dis::reader_ct_arg::NUM_SPLITS);
    constexpr uint32_t cb_q = get_compile_time_arg_val(dis::reader_ct_arg::CB_Q);
    constexpr uint32_t cb_w = get_compile_time_arg_val(dis::reader_ct_arg::CB_W);
    constexpr uint32_t cb_w_in = get_compile_time_arg_val(dis::reader_ct_arg::CB_W_IN);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(dis::reader_ct_arg::CB_K_RM);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(dis::reader_ct_arg::CB_NEGINF);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(dis::reader_ct_arg::CB_META);
    constexpr uint32_t cb_start = get_compile_time_arg_val(dis::reader_ct_arg::CB_START);
    constexpr uint32_t cb_plan = get_compile_time_arg_val(dis::reader_ct_arg::CB_PLAN);

    constexpr auto q_args = TensorAccessorArgs<dis::reader_ct_arg::END, dis::reader_common_arg::END>();
    constexpr auto w_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto keys_args =
        TensorAccessorArgs<w_args.next_compile_time_args_offset(), w_args.next_common_runtime_args_offset()>();
    constexpr auto start_args =
        TensorAccessorArgs<keys_args.next_compile_time_args_offset(), keys_args.next_common_runtime_args_offset()>();

    const uint32_t q_addr = get_common_arg_val<uint32_t>(dis::reader_common_arg::Q_ADDRESS);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(dis::reader_common_arg::W_ADDRESS);
    const uint32_t keys_addr = get_common_arg_val<uint32_t>(dis::reader_common_arg::KEYS_ADDRESS);
    const uint32_t start_addr = get_common_arg_val<uint32_t>(dis::reader_common_arg::START_ADDRESS);
    const uint32_t num_entries = get_common_arg_val<uint32_t>(dis::reader_common_arg::NUM_ENTRIES);
    const uint32_t first_q_tile = get_arg_val<uint32_t>(dis::core_arg::FIRST_Q_TILE);
    const uint32_t q_tile_stride = get_arg_val<uint32_t>(dis::core_arg::Q_TILE_STRIDE);
    const uint32_t num_q_tiles = get_arg_val<uint32_t>(dis::core_arg::NUM_Q_TILES);
    const uint32_t split = get_arg_val<uint32_t>(dis::core_arg::SPLIT);

    constexpr uint32_t tile_bytes = TILE_HW * sizeof(uint16_t);
    constexpr uint32_t head_dim_tiles = head_dim / TILE_WIDTH;
    constexpr uint32_t token_tiles = num_tokens / TILE_HEIGHT;
    constexpr uint32_t head_tiles = num_heads / TILE_WIDTH;
    constexpr uint32_t key_row_bytes = head_dim * sizeof(uint16_t);
    constexpr uint32_t step_rows = dis::STEP_KEY_TILES * TILE_HEIGHT;
    constexpr uint32_t steps_per_block = dis::TOPK / step_rows;

    Noc noc;
    experimental::CB q_cb(cb_q), w_cb(cb_w), w_in_cb(cb_w_in), k_cb(cb_k_rm), neginf_cb(cb_neginf);
    experimental::CB meta_cb(cb_meta), start_cb(cb_start), plan_cb(cb_plan);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto keys = TensorAccessor(keys_args, keys_addr);

    start_cb.reserve_back(1);
    const uint32_t start =
        trace_metadata::read_metadata_scalar_u32(noc, start_args, start_addr, start_cb.get_write_ptr());

    // The all -inf chunk that seeds the survivors of queries that see nothing in this split (two tiles, a
    // CAND_TOPK-wide chunk).
    neginf_cb.reserve_back(dis::NEGINF_TILES);
    {
        auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(neginf_cb.get_write_ptr());
        for (uint32_t i = 0; i < dis::NEGINF_TILES * tile_bytes / sizeof(uint32_t); ++i) {
            p[i] = 0xFF80FF80u;
        }
    }
    neginf_cb.push_back(dis::NEGINF_TILES);

    for (uint32_t unit = 0; unit < num_q_tiles; ++unit) {
        const uint32_t q_tile = first_q_tile + unit * q_tile_stride;

        // Query s of the tile sees entries [0, limit_s); limits never exceed the rows idx_keys holds.
        meta_cb.reserve_back(1);
        auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_cb.get_write_ptr());
        uint32_t last_limit = 0;
        for (uint32_t s = 0; s < dis::ROWS; ++s) {
            const uint32_t limit = (start + q_tile * TILE_HEIGHT + s + 1) / rate;
            last_limit = limit < num_entries ? limit : num_entries;
            meta[dis::meta_word::LIMITS + s] = last_limit;
        }
        const uint32_t tile_blocks = (last_limit + dis::TOPK - 1) / dis::TOPK;
        const uint32_t first_block = dis::split_first_block(split, tile_blocks, num_splits);
        const uint32_t end_block = dis::split_first_block(split + 1, tile_blocks, num_splits);
        meta[dis::meta_word::NUM_BLOCKS] = end_block - first_block;
        meta[dis::meta_word::FIRST_ENTRY] = first_block * dis::TOPK;
        meta[dis::meta_word::TILE_BLOCKS] = tile_blocks;
        // The writer follows the same schedule (it moves survivors, and stages block maxima in candidate mode).
        plan_cb.reserve_back(1);
        auto* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_cb.get_write_ptr());
        for (uint32_t i = 0; i < dis::meta_word::END; ++i) {
            plan[i] = meta[i];
        }
        plan_cb.push_back(1);
        meta_cb.push_back(1);

        // q [1, H, T, D] tiles of this query tile, head-major: [head][dim tile].
        q_cb.reserve_back(num_heads * head_dim_tiles);
        uint32_t q_ptr = q_cb.get_write_ptr();
        for (uint32_t h = 0; h < num_heads; ++h) {
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                noc.async_read(
                    q,
                    CoreLocalMem<uint32_t>(q_ptr),
                    tile_bytes,
                    {.page_id = (h * token_tiles + q_tile) * head_dim_tiles + d},
                    {});
                q_ptr += tile_bytes;
            }
        }

        // Gates [1, 1, T, H]: one tile per 32 heads, spread to one tile per head holding its gate in column 0
        // (the broadcast-column operand of the gate multiply). cb_w_in is reader-private scratch, never pushed.
        w_in_cb.reserve_back(head_tiles);
        const uint32_t w_in_addr = w_in_cb.get_write_ptr();
        for (uint32_t t = 0; t < head_tiles; ++t) {
            noc.async_read(
                w,
                CoreLocalMem<uint32_t>(w_in_addr + t * tile_bytes),
                tile_bytes,
                {.page_id = q_tile * head_tiles + t},
                {});
        }
        noc.async_read_barrier();
        q_cb.push_back(num_heads * head_dim_tiles);

        w_cb.reserve_back(num_heads);
        {
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_in_addr);
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_cb.get_write_ptr());
            for (uint32_t h = 0; h < num_heads; ++h) {
                const uint32_t src_tile = (h / TILE_WIDTH) * TILE_HW;
                const uint32_t col = h % TILE_WIDTH;
                for (uint32_t r = 0; r < TILE_HEIGHT; ++r) {
                    const uint32_t face_row = (r / FACE_HEIGHT) * 2 * FACE_HW + (r % FACE_HEIGHT) * FACE_WIDTH;
                    dst[h * TILE_HW + face_row] =
                        src[src_tile + face_row + (col / FACE_WIDTH) * FACE_HW + col % FACE_WIDTH];
                }
            }
        }
        w_cb.push_back(num_heads);

        // idx_keys rows of the split's blocks, one step of step_rows rows at a time. Rows past the buffer are not
        // read; their scores are never selected.
        for (uint32_t b = first_block; b < end_block; ++b) {
            for (uint32_t step = 0; step < steps_per_block; ++step) {
                const uint32_t row0 = b * dis::TOPK + step * step_rows;
                k_cb.reserve_back(step_rows);
                const uint32_t k_ptr = k_cb.get_write_ptr();
                const uint32_t rows = row0 >= num_entries              ? 0
                                      : num_entries - row0 < step_rows ? num_entries - row0
                                                                       : step_rows;
                for (uint32_t r = 0; r < rows; ++r) {
                    noc.async_read(
                        keys,
                        CoreLocalMem<uint32_t>(k_ptr + r * key_row_bytes),
                        key_row_bytes,
                        {.page_id = row0 + r},
                        {});
                }
                noc.async_read_barrier();
                k_cb.push_back(step_rows);
            }
        }
    }
}
