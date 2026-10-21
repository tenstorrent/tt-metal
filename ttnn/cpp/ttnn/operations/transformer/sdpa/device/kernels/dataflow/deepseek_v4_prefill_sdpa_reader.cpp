// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// deepseek_v4_prefill_sdpa reader. Per query token it builds the token's key list in cb_idx: the valid rows of its
// cache_idx row (sentinels may sit anywhere; they are compacted out), then its sliding-window rows of window_kv,
// tagged WINDOW_ROW. The list feeds sparse_sdpa's compute unchanged: valid keys form a prefix, so only
// ceil(valid / k_chunk) chunks are gathered and the last chunk's tail is masked. Each chunk's gather is split across
// both NoCs: the writer gathers rows [0, half) of it, this kernel rows [half, valid).

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "deepseek_v4_prefill_sdpa_gather.hpp"
#include "metadata_scalar_read.hpp"

namespace ds = deepseek_v4_prefill_sdpa;

constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(ds::reader_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(ds::reader_ct_arg::S);
    constexpr uint32_t topk = get_compile_time_arg_val(ds::reader_ct_arg::TOPK);
    constexpr uint32_t list_capacity = get_compile_time_arg_val(ds::reader_ct_arg::LIST_CAPACITY);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(ds::reader_ct_arg::K_CHUNK);
    constexpr uint32_t k_dim = get_compile_time_arg_val(ds::reader_ct_arg::K_DIM);
    constexpr uint32_t window_size = get_compile_time_arg_val(ds::reader_ct_arg::WINDOW);
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(ds::reader_ct_arg::CB_Q_RM);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(ds::reader_ct_arg::CB_K_RM);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(ds::reader_ct_arg::CB_MASK_PART);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(ds::reader_ct_arg::CB_IDX);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(ds::reader_ct_arg::CB_CTRL);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(ds::reader_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(ds::reader_ct_arg::CB_KACK);
    constexpr uint32_t cb_start = get_compile_time_arg_val(ds::reader_ct_arg::CB_START);
    constexpr uint32_t compress_ratio = get_compile_time_arg_val(ds::reader_ct_arg::COMPRESS_RATIO);
    constexpr bool empty_tail = get_compile_time_arg_val(ds::reader_ct_arg::EMPTY_TAIL) != 0;
    static_assert(topk + window_size <= list_capacity);

    constexpr auto q_args = TensorAccessorArgs<ds::reader_ct_arg::END, ds::reader_common_arg::END>();
    constexpr auto window_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    // cache_kv's length rides on common runtime args, so a longer cache reuses the program.
    constexpr auto cache_args = TensorAccessorArgs<
        window_args.next_compile_time_args_offset(),
        window_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<cache_args.next_compile_time_args_offset(), cache_args.next_common_runtime_args_offset()>();
    constexpr auto start_args =
        TensorAccessorArgs<idx_args.next_compile_time_args_offset(), idx_args.next_common_runtime_args_offset()>();

    const uint32_t q_addr = get_common_arg_val<uint32_t>(ds::reader_common_arg::Q_ADDRESS);
    const uint32_t window_addr = get_common_arg_val<uint32_t>(ds::reader_common_arg::WINDOW_ADDRESS);
    const uint32_t cache_addr = get_common_arg_val<uint32_t>(ds::reader_common_arg::CACHE_ADDRESS);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(ds::reader_common_arg::INDICES_ADDRESS);
    const uint32_t start_addr = get_common_arg_val<uint32_t>(ds::reader_common_arg::START_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t row_bytes = k_dim * sizeof(uint16_t);  // bf16 Q and K rows
    // Compute runs on whole tiles of heads; rows [H, H_pad) are zero queries whose outputs the writer drops.
    constexpr uint32_t H_pad =
        (H + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT * tt::constants::TILE_HEIGHT;
    constexpr uint32_t idx_row_bytes = topk * sizeof(uint32_t);

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    experimental::CB kreq_cb(cb_kreq), kack_cb(cb_kack), start_cb(cb_start);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto window = TensorAccessor(window_args, window_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // The chunk's absolute start position comes from a device tensor, so a trace replays with a new position.
    start_cb.reserve_back(1);
    const uint32_t start =
        trace_metadata::read_metadata_scalar_u32(noc, start_args, start_addr, start_cb.get_write_ptr());
    // window_kv row r holds position start - window_size + r; rows before position 0 hold no token.
    const uint32_t first_real_row = empty_tail ? window_size : (start >= window_size ? 0 : window_size - start);

    // The key list, shared with the writer (reserved once, reused per token).
    idx_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // Q: H head-rows -> cb_q_rm. Q is [1, H, S, K_DIM] row-major: row (h, tok) = page h * S + tok.
        q_cb.reserve_back(H_pad);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_read(q, q_cb, row_bytes, {.page_id = h * S + tok}, {.offset_bytes = h * row_bytes});
        }
        if constexpr (H_pad > H) {
            noc.async_write_zeros(q_cb, (H_pad - H) * row_bytes, {.offset_bytes = H * row_bytes});
            noc.write_zeros_l1_barrier();
        }

        uint32_t nv = 0;
        if constexpr (topk > 0) {
            noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
            noc.async_read_barrier();
            // Rows past the query's newest entry are dropped like sentinels (which are past every entry).
            uint32_t visible = ds::SENTINEL;
            if constexpr (compress_ratio > 0) {
                visible = (start + tok + 1) / compress_ratio;
            }
            for (uint32_t j = 0; j < topk; ++j) {
                const uint32_t row = list[j];
                if (row < visible) {
                    list[nv++] = row;
                }
            }
        } else {
            noc.async_read_barrier();
        }
        q_cb.push_back(H_pad);

        // Query tok sees window rows tok + 1 .. tok + window_size; the last one is the token itself.
        const uint32_t window_lo = tok + 1 > first_real_row ? tok + 1 : first_real_row;
        for (uint32_t row = window_lo; row <= tok + window_size; ++row) {
            list[nv++] = row | ds::WINDOW_ROW;
        }
        const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;

        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[sparse_sdpa::control_message::ACTIVE_CHUNKS] = n_active;
            cp[sparse_sdpa::control_message::VALID_KEYS] = nv;
        }
        ctrl_cb.push_back(1);

        for (uint32_t chunk = 0; chunk < n_active; ++chunk) {
            const uint32_t base = chunk * k_chunk;
            const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;  // (0, k_chunk]

            k_cb.reserve_back(k_chunk);
            const uint32_t k_write_ptr = k_cb.get_write_ptr();
            const uint32_t half = valid >> 1;
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                request[sparse_sdpa::gather_request::BASE] = base;
                request[sparse_sdpa::gather_request::SPLIT] = half;
                request[sparse_sdpa::gather_request::IS_LAST] = chunk == n_active - 1;
                request[sparse_sdpa::gather_request::DST_L1] = k_write_ptr;
            }
            kreq_cb.push_back(1);
            ds::gather_rows<ds::K_TRID_RING>(noc, window, cache, k_write_ptr, list, base, half, valid, row_bytes);
            kack_cb.wait_front(1);  // the writer's half landed in cb_k_rm
            kack_cb.pop_front(1);
            // Masked keys score a defined 0 before compute adds -inf to them.
            if (valid < k_chunk) {
                noc.async_write_zeros(k_cb, (k_chunk - valid) * row_bytes, {.offset_bytes = valid * row_bytes});
                noc.write_zeros_l1_barrier();
            }
            k_cb.push_back(k_chunk);

            // Only the last chunk can be partial. When its boundary lands mid-tile, row 0 of this tile is -inf
            // from column valid % 32 on (compute adds it row-broadcast; later tiles get the all-(-inf) tile).
            if (chunk == n_active - 1 && (valid % tt::constants::TILE_WIDTH) != 0) {
                mask_part_cb.reserve_back(1);
                {
                    const uint32_t first_masked_col = valid % tt::constants::TILE_WIDTH;
                    volatile tt_l1_ptr uint16_t* mask_tile =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
                    for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                        const uint32_t elem_off = (col < tt::constants::FACE_WIDTH)
                                                      ? col
                                                      : (tt::constants::FACE_HW + col - tt::constants::FACE_WIDTH);
                        mask_tile[elem_off] = (col >= first_masked_col) ? neg_inf_bf16 : (uint16_t)0;
                    }
                }
                mask_part_cb.push_back(1);
            }
        }
    }
}
