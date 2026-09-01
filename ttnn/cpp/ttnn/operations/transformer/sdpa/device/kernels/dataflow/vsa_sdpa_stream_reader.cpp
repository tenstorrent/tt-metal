// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v2) reader (BRISC): owns the block stream schedule. Builds per-resident-row
// membership bitmaps from the index rows, then walks blocks in ascending id order; for each block
// any resident row lists, it takes a stream-slot credit, fetches the V tiles (NCRISC fetches K in
// parallel via the kreq/kack pair), builds the ragged partial-column mask when needed, and emits
// one VISIT ctrl message per listing row (the last one carries the slot-free flag). After the
// stream, one FLUSH per resident row. Rows are processed in passes of at most R_MAX.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"
#include "dataflow_common.hpp"

constexpr uint32_t sentinel = 0xFFFFFFFFu;

constexpr uint32_t MSG_VISIT = 0;
constexpr uint32_t MSG_FLUSH = 1;
constexpr uint32_t FLAG_IS_FIRST = 1u << 8;
constexpr uint32_t FLAG_LAST_OF_BLOCK = 1u << 9;
constexpr uint32_t FLAG_HAS_VMASK = 1u << 10;

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);            // index row width (entries)
    constexpr uint32_t n_kv_blocks = get_compile_time_arg_val(1);  // T / block_size
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(2);    // S / 64 per head
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t R_MAX = get_compile_time_arg_val(4);
    constexpr uint32_t stream_depth = get_compile_time_arg_val(5);
    constexpr uint32_t v_tiles_per_block = get_compile_time_arg_val(6);
    constexpr uint32_t v_head_stride = get_compile_time_arg_val(7);  // tiles per head in v
    constexpr uint32_t idx_row_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t counts_row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t v_tile_bytes = get_compile_time_arg_val(10);

    constexpr uint32_t cb_v_stream = get_compile_time_arg_val(11);
    constexpr uint32_t cb_idxrow = get_compile_time_arg_val(12);   // one index row scratch
    constexpr uint32_t cb_counts = get_compile_time_arg_val(13);   // counts row, resident
    constexpr uint32_t cb_bitmap = get_compile_time_arg_val(14);   // R_MAX x ceil(KVB/32) words
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(15);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(16);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(17);
    constexpr uint32_t cb_free = get_compile_time_arg_val(18);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(19);

    constexpr auto v_args = TensorAccessorArgs<20, 0>();
    constexpr auto idx_args =
        TensorAccessorArgs<v_args.next_compile_time_args_offset(), v_args.next_common_runtime_args_offset()>();
    constexpr auto counts_args =
        TensorAccessorArgs<idx_args.next_compile_time_args_offset(), idx_args.next_common_runtime_args_offset()>();

    const uint32_t v_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t row_stride = get_arg_val<uint32_t>(5);
    const uint32_t row_count = get_arg_val<uint32_t>(6);

    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint32_t bitmap_words = (n_kv_blocks + 31) / 32;

    Noc noc;
    experimental::CB v_cb(cb_v_stream), idx_cb(cb_idxrow), counts_cb(cb_counts);
    experimental::CB ctrl_cb(cb_ctrl), kreq_cb(cb_kreq), kack_cb(cb_kack), free_cb(cb_free);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);

    counts_cb.reserve_back(1);
    noc.async_read(counts, counts_cb, counts_row_bytes, {.page_id = 0}, {.offset_bytes = 0});
    volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr());

    idx_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    experimental::CB bitmap_cb(cb_bitmap);
    bitmap_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* bitmaps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bitmap_cb.get_write_ptr());
    noc.async_read_barrier();  // counts resident

    const uint32_t v_base = head * v_head_stride;
    uint32_t next_slot = 0;

    for (uint32_t pass_base = 0; pass_base < row_count; pass_base += R_MAX) {
        const uint32_t pass_rows = (row_count - pass_base < R_MAX) ? (row_count - pass_base) : R_MAX;

        // membership bitmaps for this pass's rows
        for (uint32_t r = 0; r < pass_rows; ++r) {
            volatile tt_l1_ptr uint32_t* bm = bitmaps + r * bitmap_words;
            for (uint32_t wd = 0; wd < bitmap_words; ++wd) {
                bm[wd] = 0;
            }
            const uint32_t q_tile = row_start + (pass_base + r) * row_stride;
            noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = head * n_q_tiles + q_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            for (uint32_t e = 0; e < W; ++e) {
                const uint32_t b = idx_ptr[e];
                if (b == sentinel) {
                    break;
                }
                ASSERT(b < n_kv_blocks);
                bm[b >> 5] |= (1u << (b & 31));
            }
        }

        uint32_t first_pending = (1u << pass_rows) - 1u;  // rows awaiting their first visit (R_MAX <= 32)
        uint32_t listing[32];

        for (uint32_t b = 0; b < n_kv_blocks; ++b) {
            uint32_t n_listing = 0;
            const uint32_t wd = b >> 5;
            const uint32_t bit = 1u << (b & 31);
            for (uint32_t r = 0; r < pass_rows; ++r) {
                if (bitmaps[r * bitmap_words + wd] & bit) {
                    listing[n_listing++] = r;
                }
            }
            if (n_listing == 0) {
                continue;
            }

            free_cb.wait_front(1);  // a stream slot is free
            free_cb.pop_front(1);
            const uint32_t slot = next_slot;
            next_slot = (next_slot + 1 == stream_depth) ? 0 : next_slot + 1;

            // NCRISC fetches K for this block in parallel.
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* rq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    kreq_cb.get_write_ptr());
                rq[0] = b;
                rq[1] = slot;
            }
            kreq_cb.push_back(1);

            const uint32_t count = counts_ptr[b];
            const bool ragged = count < block_size;
            const bool needs_vmask = ragged && (count % keys_per_tile) != 0;
            if (needs_vmask) {
                constexpr uint32_t mask_tile_bytes = get_tile_size(cb_vmask);
                experimental::CB vmask_cb(cb_vmask);
                vmask_cb.reserve_back(1);
                fill_vertical_tile_bf16<mask_tile_bytes>(noc, cb_vmask, 0, count % keys_per_tile);
                vmask_cb.push_back(1);
            }

            {
                sparse_sdpa_msa::TridRing ring{noc};
                const uint32_t v_tile0 = v_base + b * v_tiles_per_block;
                for (uint32_t i = 0; i < v_tiles_per_block; ++i) {
                    ring.read(v, v_cb, v_tile_bytes, v_tile0 + i, (slot * v_tiles_per_block + i) * v_tile_bytes);
                }
                ring.drain();
            }
            kack_cb.wait_front(1);  // K landed
            kack_cb.pop_front(1);

            for (uint32_t i = 0; i < n_listing; ++i) {
                const uint32_t r = listing[i];
                uint32_t flags = MSG_VISIT;
                if (first_pending & (1u << r)) {
                    flags |= FLAG_IS_FIRST;
                    first_pending &= ~(1u << r);
                }
                if (i + 1 == n_listing) {
                    flags |= FLAG_LAST_OF_BLOCK;
                }
                if (needs_vmask) {
                    flags |= FLAG_HAS_VMASK;
                }
                ctrl_cb.reserve_back(1);
                {
                    volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        ctrl_cb.get_write_ptr());
                    cp[0] = flags;
                    cp[1] = r;
                    cp[2] = slot;
                    cp[3] = count;
                }
                ctrl_cb.push_back(1);
            }
        }

        // every row has at least one valid block, so first_pending must be clear by now
        ASSERT(first_pending == 0);
        for (uint32_t r = 0; r < pass_rows; ++r) {
            ctrl_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    ctrl_cb.get_write_ptr());
                cp[0] = MSG_FLUSH;
                cp[1] = r;
                cp[2] = 0;
                cp[3] = 0;
            }
            ctrl_cb.push_back(1);
        }
    }
}
