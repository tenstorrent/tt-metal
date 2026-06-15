// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa reader: per token, gather Q rows + per-chunk indexed K rows (sentinel suffix zero-filled),
// plus the per-token partial-boundary mask tile. (The persistent scaler/col-identity/-inf tiles are built
// by the lighter-loaded writer.) Uses the object-based NoC + circular-buffer APIs.
//
// Sentinels are a contiguous tail (producer contract): nv = first-sentinel index = valid-key count, so only
// ceil(nv/k_chunk) chunks are active; all-sentinel chunks are skipped (no read/fill/compute), compute learns
// nv + the active count via cb_ctrl. Dense-ish tokens (n_active >= ring_min_active) gather K through a NoC
// trid ring that bounds outstanding reads/core to fight DRAM congestion.

#include <stdint.h>
#include <tt-metalium/constants.hpp>  // tt::constants::TILE_HEIGHT
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include <tools/profiler/kernel_profiler.hpp>  // DeviceZoneScopedN

constexpr uint32_t sentinel = 0xFFFFFFFFu;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t topk = get_compile_time_arg_val(2);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t k_trids = get_compile_time_arg_val(4);          // NoC trid-ring depth (0 = disabled)
    constexpr uint32_t ring_min_active = get_compile_time_arg_val(5);  // ring only if n_active >= this
    constexpr uint32_t k_dim = get_compile_time_arg_val(6);            // q/kv head dim (from the tensor)

    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's reader CB-arg block (compile args 7..11).
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(8);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(9);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(11);

    constexpr auto q_args = TensorAccessorArgs<12>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t tok_start = get_arg_val<uint32_t>(3);
    const uint32_t tok_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_bytes = k_dim * 2;  // K/Q row
    constexpr uint32_t idx_row_bytes = topk * 4;

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // Reader-internal scratch for one token's index row (reserved once, reused).
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // Q: H head-rows -> cb_q_rm.  Q is [1,H,S,576] row-major: row (h,tok) = page h*S+tok.
        q_cb.reserve_back(H);
        {
            DeviceZoneScopedN("rdr_Q");
            for (uint32_t h = 0; h < H; ++h) {
                noc.async_read(q, q_cb, row_bytes, {.page_id = h * S + tok}, {.offset_bytes = h * row_bytes});
            }
            noc.async_read_barrier();
        }
        q_cb.push_back(H);

        // indices row for this token (page = tok) -> idx_cb scratch
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // binary-search the first sentinel -> nv (valid-key count); only ceil(nv/k_chunk) chunks are active.
        uint32_t nv = topk;
        {
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (idx_ptr[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            nv = lo == 0 ? 1 : lo;  // contract guarantees >=1 valid; guard anyway
        }
        const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;

        // hand the active chunk count + valid-key count to compute (it can't issue DRAM reads to derive
        // them). nv lets compute place the boundary mask (which key tiles of the last chunk to -inf).
        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[0] = n_active;
            cp[1] = nv;
        }
        ctrl_cb.push_back(1);

        for (uint32_t c = 0; c < n_active; ++c) {
            const uint32_t base = c * k_chunk;
            const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;  // (0, k_chunk]

            // K chunk rows -> cb_k_rm: read the `valid` real keys, zero-fill the sentinel suffix.
            k_cb.reserve_back(k_chunk);
            {
                DeviceZoneScopedN("rdr_Kchunk");  // the per-chunk indexed K gather (dominant cost)
                auto plain_gather = [&]() {
                    for (uint32_t j = 0; j < valid; ++j) {
                        noc.async_read(
                            kv, k_cb, row_bytes, {.page_id = idx_ptr[base + j]}, {.offset_bytes = j * row_bytes});
                    }
                    noc.async_read_barrier();
                };
                // Sparse tokens (and k_trids==0) gather plainly: cores desync naturally, no congestion to
                // recover. Dense-ish tokens use a trid ring: tag read j with trid (j % k_trids)+1 and barrier
                // on a trid before reusing it -> bounds outstanding reads/core, capping NoC/DRAM congestion.
                if constexpr (k_trids == 0) {
                    plain_gather();
                } else if (n_active < ring_min_active) {
                    plain_gather();
                } else {
                    for (uint32_t j = 0; j < valid; ++j) {
                        const uint32_t trid = (j % k_trids) + 1;
                        if (j >= k_trids) {
                            experimental::async_read_barrier_with_trid(noc, trid);  // free this slot
                        }
                        experimental::set_read_trid(noc, trid);
                        noc.async_read(
                            kv, k_cb, row_bytes, {.page_id = idx_ptr[base + j]}, {.offset_bytes = j * row_bytes});
                    }
                    const uint32_t to_drain = (valid < k_trids) ? valid : k_trids;
                    for (uint32_t d = 0; d < to_drain; ++d) {
                        experimental::async_read_barrier_with_trid(noc, ((valid - to_drain + d) % k_trids) + 1);
                    }
                    experimental::set_read_trid(noc, 0);  // restore untagged
                }
                // Zero-fill the sentinel suffix: masked-key scores become a defined 0 (compute adds -inf).
                if (valid < k_chunk) {
                    noc.async_write_zeros(k_cb, (k_chunk - valid) * row_bytes, {.offset_bytes = valid * row_bytes});
                    noc.write_zeros_l1_barrier();
                }
            }
            k_cb.push_back(k_chunk);

            // Boundary mask: only the LAST active chunk can have sentinels. Its key tile that straddles
            // `valid` (tile valid/32) needs cols >= valid%32 set to -inf; the tiles after it are fully
            // masked (compute bcast-adds the persistent cb_neginf there). Build the single partial-boundary
            // tile here only when the boundary lands mid-tile; row 0 only (compute adds it row-broadcast).
            if (c == n_active - 1 && (valid % tt::constants::TILE_WIDTH) != 0) {
                mask_part_cb.reserve_back(1);
                {
                    DeviceZoneScopedN("rdr_mask");  // build the single partial-boundary mask tile
                    const uint32_t b = valid % tt::constants::TILE_WIDTH;  // first masked col within the tile
                    volatile tt_l1_ptr uint16_t* p =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
                    for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                        const uint32_t off = (col < 16) ? col : (256 + col - 16);  // face0/face1 row 0
                        p[off] = (col >= b) ? neg_inf_bf16 : (uint16_t)0;
                    }
                }
                mask_part_cb.push_back(1);
            }
        }
    }
}
