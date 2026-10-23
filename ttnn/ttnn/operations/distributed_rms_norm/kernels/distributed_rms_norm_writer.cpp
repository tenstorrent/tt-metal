// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm writer (compute cores, BRISC / NoC1) — op_design.md -> Dataflow Strategy.
//
// γ: for each valid tile of this core's W-slice, read the tile's row-0 byte prefix (face 0 row 0 .. face 1 row 0; base
// op GAMMA_DIRECT_READ = PREFIX) straight into its cb_gamma_tiles page. ROW_MAJOR γ sticks (GAMMA_RM; per device
// (1, 1, ceil(W/32), 32), stick r = row 0 of γ tile r): stick w0 + j is read whole (one aligned page) into page j's
// face 0 row 0, then its elements 16..31 are moved to face 1 row 0 (a 32 / 64 B L1 word copy: a NoC read of the
// stick's second half would start 32 B into a DRAM page, off the DRAM read alignment). GAMMA_RESIDENT: the WS-page
// window is pushed once in a prologue (resident for the kernel). Streamed (an L1 fallback for wide shards): per block,
// per chunk c, C pages are pushed ahead of the chunk's output quanta (always C pages, so the ring never straddles). The
// BroadcastDim::Row multiply reads row 0 only.
// store_block: per C-tile chunk, per tile-row of the block (compute's production order): wait C pages of cb_output,
// write the slice's valid tiles (page tile_row * Wt + w0 + j), flush, pop C (pad tiles of a ragged slice are never
// stored). OUT_SHARD: cb_output IS the output shard and compute
// packs straight into it — nothing to store.
// Raw NoC (op_design.md -> "Helpers considered and rejected"): no kernel_lib helper writes TILE pages.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t num_waves = get_compile_time_arg_val(1);
    constexpr uint32_t WS = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t nch = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr bool has_gamma = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t gamma_prefix_bytes = get_compile_time_arg_val(16);
    constexpr bool out_shard = get_compile_time_arg_val(19) != 0;
    constexpr bool gamma_resident = get_compile_time_arg_val(20) != 0;
    constexpr bool gamma_rm = get_compile_time_arg_val(23) != 0;  // γ given as ROW_MAJOR sticks
    constexpr auto output_args = TensorAccessorArgs<28>();        // after the 28 common CT args (host COMMON_CT_ARGS)
    constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_gamma_tiles = 9;
    constexpr uint32_t cb_output = 16;
    constexpr uint32_t T_out = get_tile_size(cb_output);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t w0 = get_arg_val<uint32_t>(4);
    const uint32_t V = get_arg_val<uint32_t>(5);

    const auto output = TensorAccessor(output_args, out_addr);
    // γ tiles [j0, j1) of the slice -> `pages` pages of cb_gamma_tiles (row-0 prefixes only).
    auto load_gamma = [&](uint32_t j0, uint32_t j1, uint32_t pages) {
        if constexpr (has_gamma) {
            const auto gamma = TensorAccessor(gamma_args, gamma_addr);
            constexpr uint32_t T_g = get_tile_size(cb_gamma_tiles);
            cb_reserve_back(cb_gamma_tiles, pages);
            const uint32_t l1 = get_write_ptr(cb_gamma_tiles);
            for (uint32_t j = j0; j < j1; ++j) {
                // TILE: the tile's row-0 prefix; RM: the whole stick (gamma_prefix_bytes = 32 elements, host).
                noc_async_read(gamma.get_noc_addr(w0 + j), l1 + (j - j0) * T_g, gamma_prefix_bytes);
            }
            noc_async_read_barrier();
            if constexpr (gamma_rm) {
                // Stick elements 16..31 (face 0 row 1, as read) -> face 1 row 0. Face 0 row 1 keeps stale values:
                // the BroadcastDim::Row multiply reads row 0 only.
                constexpr uint32_t half_words = gamma_prefix_bytes / 2 / 4;
                constexpr uint32_t face_words = T_g / 4 / 4;
                invalidate_l1_cache();
                for (uint32_t j = j0; j < j1; ++j) {
                    volatile tt_l1_ptr uint32_t* page =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + (j - j0) * T_g);
                    for (uint32_t k = 0; k < half_words; ++k) {
                        page[face_words + k] = page[half_words + k];
                    }
                }
            }
            cb_push_back(cb_gamma_tiles, pages);
        }
    };

    if constexpr (has_gamma && gamma_resident) {
        load_gamma(0, V, WS);
    }
    if constexpr (out_shard && (!has_gamma || gamma_resident)) {
        return;  // compute packs into the output shard (zero-copy): nothing left to move
    }

    for (uint32_t w = 0; w < num_waves; ++w) {
        const uint32_t first = w * R;
        const uint32_t R_b = rows > first ? (rows - first < R ? rows - first : R) : 0;
        // Quanta arrive in compute's order: chunk-outer, tile-row-inner (scale_by_inv_rms_block runs grid(R_b, C)
        // once per chunk c, pushing one C-page quantum per tile-row). The store loop must walk the same order.
        if (R_b == 0) {
            continue;
        }
        for (uint32_t c = 0; c < nch; ++c) {
            const uint32_t j0 = c * C;
            const uint32_t j1 = (j0 + C) < V ? j0 + C : V;
            if constexpr (has_gamma && !gamma_resident) {
                load_gamma(j0, j1, C);  // chunk c's γ, ahead of its output quanta
            }
            if constexpr (out_shard) {
                continue;
            }
            for (uint32_t t = 0; t < R_b; ++t) {
                const uint32_t page0 = (start_row + first + t) * Wt + w0;
                {
                    DeviceZoneScopedN("wr_wait_out");
                    cb_wait_front(cb_output, C);
                }
                DeviceZoneScopedN("wr_store");
                const uint32_t l1 = get_read_ptr(cb_output);
                for (uint32_t j = j0; j < j1; ++j) {
                    noc_async_write_tile(page0 + j, output, l1 + (j - j0) * T_out);
                }
                noc_async_writes_flushed();  // the CB slot may be reused once its bytes left L1
                cb_pop_front(cb_output, C);
            }
        }
    }
    noc_async_write_barrier();  // every output write acknowledged before the program ends
}
