// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute-core reader (RISCV_1, NoC0). Core (q, m) owns block m of slice q (block_h x block_w tiles in `landing`).
//   prepare_gamma_block   gamma sticks q*Wl_t .. +block_w (64 B) -> row 0 of block_w tiles (cb_gamma); the sticks come
//                         from the hub's multicast (GAMMA_HUB_MCAST: wait gamma_go) or from a per-core DRAM read
//   load_own_block        (q == p) DRAM x -> cb_x_block (the landing shard), one barrier, then own_loaded++ on both
//                         ports of the block's link (they read the own slice from here, L1 -> fabric)
//   stats -> tiles        as each slot's bit appears in stats_ready (the hub's per-slot multicast), col 0 of tile q'*h
//   + r of cb_partials =
//                         stats[q'][m*block_h + r] (32 fp32), for every slice q'
//   landed x              (q != p) per chunk k: wait landed >= k + 1, push chunk_tiles. Zero-copy: the fabric already
//                         wrote the bytes into the landing shard.
//
// Raw dataflow (no helper covers these): gamma stick -> tile row 0 (tilize_helpers assume one page per full row, a
// row broadcast needs only row 0); stats column -> col 0 of an fp32 tile (no helper moves a tile column).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "all_gather_rms_norm_kimi_common.hpp"

using namespace agrn;

constexpr uint32_t C = NUM_GEO_CT;
constexpr uint32_t cb_x_block = get_compile_time_arg_val(C + 0);
constexpr uint32_t cb_gamma = get_compile_time_arg_val(C + 1);
constexpr uint32_t cb_partials = get_compile_time_arg_val(C + 2);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(C + 3);
constexpr uint32_t sem_own_loaded = get_compile_time_arg_val(C + 4);
constexpr uint32_t sem_stats_ready = get_compile_time_arg_val(C + 5);
constexpr uint32_t sem_landed = get_compile_time_arg_val(C + 6);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(C + 7);
constexpr uint32_t gamma_page = get_compile_time_arg_val(C + 8);
constexpr uint32_t sem_gamma_go = get_compile_time_arg_val(C + 9);
constexpr bool gamma_hub_mcast = get_compile_time_arg_val(C + 10) != 0;
constexpr auto in_args = TensorAccessorArgs<C + 11>();
constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

void kernel_main() {
    // common args: in_addr, gamma_addr, stats_addr, p, row_base[grid_y], fwd port xy[L], bwd port xy[L]
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t stats_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t p = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t row_base_idx = 4;
    constexpr uint32_t ports_idx = row_base_idx + grid_y;
    const BlockId b = block_id(get_absolute_logical_x(), get_absolute_logical_y(), row_base_idx, [](uint32_t i) {
        return get_common_arg_val<uint32_t>(i);
    });
    const uint32_t n_tiles = b.h * block_w;
    const bool own = b.q == p;

    // The own block goes first: the own partial (the head of the stats-critical chain) needs only x and the
    // scaler; gamma is needed only by scale_block, after the stats round.
    cb_reserve_back(cb_x_block, n_tiles);
    if (own) {
        // ---- load_own_block: DRAM -> landing ----
        const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
        const uint32_t dst = get_write_ptr(cb_x_block);
        for (uint32_t r = 0; r < b.h; ++r) {
            const uint32_t row_page = (b.m * block_h + r) * Wl_t;
            for (uint32_t j = 0; j < block_w; ++j) {
                noc_async_read(in.get_noc_addr(row_page + j), dst + (r * block_w + j) * tile_bytes, tile_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x_block, n_tiles);
        // the ports of my link read my slice from L1 from here on
        const uint32_t l = b.m / cpl;
        const uint32_t own_loaded_addr = get_semaphore(sem_own_loaded);
        for (uint32_t d = 0; d < 2; ++d) {
            const uint32_t xy = get_common_arg_val<uint32_t>(ports_idx + d * num_links + l);
            noc_semaphore_inc(get_noc_addr(xy >> 8, xy & 0xFF, own_loaded_addr), 1);
        }
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    }

    // ---- prepare_gamma_block: stick n -> row 0 of tile j (face 0 row 0 = elements 0..15, face 1 row 0 = 16..31) ----
    {
        cb_reserve_back(cb_gamma, block_w);
        const uint32_t base = get_write_ptr(cb_gamma);
        // stick j's 64 B source: the hub's multicast in my stats-scratch tail (GAMMA_HUB_MCAST), or my own DRAM read
        // landed at the first 64 B boundary inside tile j (DRAM reads want a 64 B aligned L1 destination)
        uint32_t src0, src_stride;
        if constexpr (gamma_hub_mcast) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gamma_go)), 1);
            src0 = ((stats_addr + G * Mt * STATS_ROW_BYTES + 63) & ~63u) + b.q * Wl_t * gamma_page;
            src_stride = gamma_page;
        } else {
            const auto gamma = TensorAccessor(gamma_args, gamma_addr, gamma_page);
            const uint32_t pad = (64 - (base & 63)) & 63;
            for (uint32_t j = 0; j < block_w; ++j) {
                noc_async_read(gamma.get_noc_addr(b.q * Wl_t + j), base + j * tile_bytes + pad, gamma_page);
            }
            noc_async_read_barrier();
            src0 = base + pad;
            src_stride = tile_bytes;
        }
        for (uint32_t j = 0; j < block_w; ++j) {
            stick_to_row0(src0 + j * src_stride, base + j * tile_bytes);
        }
        cb_push_back(cb_gamma, block_w);
    }

    // ---- stats -> cb_partials: col 0 of tile (q' * h + r) = stats[q'][m * block_h + r] ----
    {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready));
        cb_reserve_back(cb_partials, G * b.h);
        const uint32_t base = get_write_ptr(cb_partials);
        // stats_ready = mask of the slots the hub has multicast so far; copy each one as its bit appears
        constexpr uint32_t all = (1u << G) - 1;
        uint32_t seen = 0;
        while (seen != all) {
            invalidate_l1_cache();
            const uint32_t fresh = *ready & ~seen;
            seen |= fresh;
            for (uint32_t q = 0; q < G; ++q) {
                if (!(fresh & (1u << q))) {
                    continue;
                }
                for (uint32_t r = 0; r < b.h; ++r) {
                    stats_row_to_col0(
                        stats_addr + (q * Mt + b.m * block_h + r) * STATS_ROW_BYTES,
                        base + (q * b.h + r) * FP32_TILE_BYTES);
                }
            }
        }
        cb_push_back(cb_partials, G * b.h);
    }

    // ---- remote block: push each chunk once its receiver says it landed ----
    if (!own) {
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_landed));
        const uint32_t n_chunks = n_tiles / chunk_tiles;
        for (uint32_t k = 0; k < n_chunks; ++k) {
            noc_semaphore_wait_min(landed, k + 1);
            cb_push_back(cb_x_block, chunk_tiles);
        }
    }
}
