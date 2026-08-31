// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa reader (forked from sparse_sdpa_msa_reader): for each (head, 64-token query tile), read the Q tiles
// and gather the row's listed KV blocks in chunks of up to m blocks. Reader and writer split each chunk's tile
// gather into upper/lower block halves. Ragged blocks (valid count < block_size) get partial-column mask tiles
// built here and consumed by compute; sentinel block ids mask the row tail.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"  // per-NoC trid-ring (shared with sparse_sdpa_msa)
#include "dataflow_common.hpp"         // fill_vertical_tile_bf16 (partial-column mask tile)

constexpr uint32_t sentinel = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);             // index row width (entries)
    constexpr uint32_t n_kv_blocks = get_compile_time_arg_val(1);   // T / block_size
    constexpr uint32_t m = get_compile_time_arg_val(2);             // blocks per chunk
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(3);     // S / 64 (per head)
    constexpr uint32_t block_size = get_compile_time_arg_val(4);    // tokens per block
    constexpr uint32_t q_tiles_per_work = get_compile_time_arg_val(5);
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(6);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(7);
    constexpr uint32_t k_half = get_compile_time_arg_val(8);  // writer gathers [0, half)
    constexpr uint32_t v_half = get_compile_time_arg_val(9);
    constexpr uint32_t k_head_stride = get_compile_time_arg_val(10);  // tiles per head in k
    constexpr uint32_t v_head_stride = get_compile_time_arg_val(11);
    constexpr uint32_t idx_row_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t counts_row_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t q_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(16);

    // CB ids match the factory's reader compile-arg block.
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(17);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(18);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(19);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(20);
    constexpr uint32_t cb_counts = get_compile_time_arg_val(21);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(22);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(23);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(24);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(25);

    constexpr auto q_args = TensorAccessorArgs<26, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    constexpr auto v_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();
    constexpr auto idx_args =
        TensorAccessorArgs<v_args.next_compile_time_args_offset(), v_args.next_common_runtime_args_offset()>();
    constexpr auto counts_args =
        TensorAccessorArgs<idx_args.next_compile_time_args_offset(), idx_args.next_common_runtime_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t work_start = get_arg_val<uint32_t>(5);
    const uint32_t work_count = get_arg_val<uint32_t>(6);

    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint32_t kreq_words = 2 + m;

    Noc noc;
    experimental::CB q_cb(cb_q_in), k_cb(cb_k_in), v_cb(cb_v_in), idx_cb(cb_idx), counts_cb(cb_counts);
    experimental::CB ctrl_cb(cb_ctrl), kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);

    // block_counts row: one page, read once, resident for the whole kernel.
    counts_cb.reserve_back(1);
    noc.async_read(counts, counts_cb, counts_row_bytes, {.page_id = 0}, {.offset_bytes = 0});
    const uint32_t counts_l1 = counts_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);

    // Reader-internal scratch for one row's block-id list (reserved once, reused).
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    noc.async_read_barrier();  // counts row landed

    for (uint32_t work = 0; work < work_count; ++work) {
        const uint32_t w = work_start + work;
        const uint32_t head = w / n_q_tiles;

        // Q tiles for this (head, q-tile): a contiguous run of Sqt*DHt pages (head-major layout).
        q_cb.reserve_back(q_tiles_per_work);
        for (uint32_t i = 0; i < q_tiles_per_work; ++i) {
            noc.async_read(
                q, q_cb, q_tile_bytes, {.page_id = w * q_tiles_per_work + i}, {.offset_bytes = i * q_tile_bytes});
        }

        // Block-id row for this (head, q-tile): page id == flat work index.
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = w}, {.offset_bytes = 0});
        noc.async_read_barrier();
        q_cb.push_back(q_tiles_per_work);

        // Binary search the first sentinel; valid blocks are a contiguous prefix.
        uint32_t n_active;
        {
            uint32_t lo = 0, hi = W;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (idx_ptr[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            n_active = lo == 0 ? 1 : lo;  // the ASSERT below traps all-sentinel rows
        }

        const uint32_t k_base = head * k_head_stride;
        const uint32_t v_base = head * v_head_stride;

        for (uint32_t chunk_start = 0; chunk_start < n_active; chunk_start += m) {
            const uint32_t n_valid = (n_active - chunk_start < m) ? (n_active - chunk_start) : m;
            const bool is_last = (chunk_start + n_valid == n_active);

            // Per-chunk control for compute: {n_valid, is_last, counts[b]...}. Compute derives the ragged
            // masks from the counts; the vmask tiles below are consumed in the same block order.
            ctrl_cb.reserve_back(1);
            uint32_t n_masks = 0;
            {
                volatile tt_l1_ptr uint32_t* cp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
                cp[0] = n_valid;
                cp[1] = is_last ? 1u : 0u;
                for (uint32_t b = 0; b < n_valid; ++b) {
                    const uint32_t block_id = idx_ptr[chunk_start + b];
                    ASSERT(block_id != sentinel);
                    ASSERT(block_id < n_kv_blocks);
                    const uint32_t count = counts_ptr[block_id];
                    cp[2 + b] = count;
                    if (count < block_size && (count % keys_per_tile) != 0) {
                        ++n_masks;
                    }
                }
            }

            // Partial-column mask tiles for the chunk's ragged blocks (columns >= count%32 get -inf in the
            // boundary key-tile; fully-padded key-tiles are stamped from cb_neginf in compute). Fixed batch of
            // m slots so every CB reservation is the same size and stays L1-contiguous: block b's mask lives at
            // slot b, slots of non-ragged blocks are never filled or read. Compute derives the same n_masks > 0
            // predicate from the counts and pops the batch.
            if (n_masks > 0) {
                constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
                experimental::CB vmask_cb(cb_vmask);
                vmask_cb.reserve_back(m);
                for (uint32_t b = 0; b < n_valid; ++b) {
                    const uint32_t count = counts_ptr[idx_ptr[chunk_start + b]];
                    if (count < block_size && (count % keys_per_tile) != 0) {
                        fill_vertical_tile_bf16<mask_tile_bytes>(noc, cb_vmask, b, count % keys_per_tile);
                    }
                }
                vmask_cb.push_back(m);
            }
            ctrl_cb.push_back(1);

            // Reserve the full fixed-size chunk (a partial chunk leaves its tail tiles unread), tell the
            // writer which blocks to co-gather, then read the upper block halves on this NoC.
            k_cb.reserve_back(m * k_tiles_per_block);
            v_cb.reserve_back(m * v_tiles_per_block);

            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                rq[0] = n_valid;
                rq[1] = is_last ? 1u : 0u;
                for (uint32_t b = 0; b < n_valid; ++b) {
                    rq[2 + b] = idx_ptr[chunk_start + b];
                }
                (void)kreq_words;
            }
            kreq_cb.push_back(1);

            sparse_sdpa_msa::TridRing ring{noc};  // K/V upper halves share one ring
            for (uint32_t b = 0; b < n_valid; ++b) {
                const uint32_t block_id = idx_ptr[chunk_start + b];
                const uint32_t k_tile0 = k_base + block_id * k_tiles_per_block;
                const uint32_t v_tile0 = v_base + block_id * v_tiles_per_block;
                const uint32_t k_off = b * k_tiles_per_block;
                const uint32_t v_off = b * v_tiles_per_block;
                for (uint32_t i = k_half; i < k_tiles_per_block; ++i) {
                    ring.read(k, k_cb, k_tile_bytes, k_tile0 + i, (k_off + i) * k_tile_bytes);
                }
                for (uint32_t i = v_half; i < v_tiles_per_block; ++i) {
                    ring.read(v, v_cb, v_tile_bytes, v_tile0 + i, (v_off + i) * v_tile_bytes);
                }
            }
            ring.drain();           // this NoC's upper halves landed
            kack_cb.wait_front(1);  // writer's lower halves landed in the same L1
            kack_cb.pop_front(1);
            k_cb.push_back(m * k_tiles_per_block);
            v_cb.push_back(m * v_tiles_per_block);
        }
    }
}
