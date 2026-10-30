// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm writer (op_design.md -> Dataflow Strategy: γ supply + store_wave).
//
// γ supply (has_gamma) — on this RISC-V (base op lamp L4: the writer is idle until the first output chunk): the
// core's γ window [w0, w0 + V) is resident in cb_gamma_tiles (Wp pages), pushed chunk by chunk (C pages); compute
// waits cumulatively and keeps it for the whole kernel. L1 fallback (gamma_resident = 0, wide-row shards): the window
// streams per (block, chunk) step through a STREAM_DEPTH * C ring, re-read from DRAM per block (base op streaming γ).
// Only row 0 of faces 0 / 1 is consumed (BroadcastDim::Row):
//   TILE γ  (1,1,1,W_local): the row-0 byte prefix of each γ tile ([0, face + 16 datums)) is read straight into its
//           page (base GAMMA_DIRECT_PREFIX; source and destination are both tile-aligned).
//   RM sticks (1,1,ceil(W/32),32): page j = γ tile-column j's 32 values; it is read into the page's first bytes
//           (row 0 of face 0 = values 0..15, the next 16 values land in face 0 row 1), then values 16..31 are moved
//           to row 0 of face 1 by 16 RISC-V word copies (no DRAM-aligned read can land there directly).
// Output (interleaved): cb_output in C-page quanta, per block, per chunk, per tile-row (the order compute packs:
//   per chunk, its waves in row order); the chunk's valid tiles are written (page tile_row * Wt + w0 + c * C + j), one
//   write flush per quantum. Sharded: cb_output IS the output shard, compute packs into it; nothing to write.
// Raw NoC / CB calls: no kernel_lib helper reads a γ row-0 prefix or relays a γ stick into a tile's row 0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_output = 16;
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
}  // namespace

void kernel_main() {
    constexpr bool has_gamma = get_compile_time_arg_val(0) != 0;
    constexpr bool gamma_is_tile = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr bool sharded = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t T_out = get_compile_time_arg_val(8);
    constexpr uint32_t T_g = get_compile_time_arg_val(9);
    constexpr uint32_t es_g = get_compile_time_arg_val(10);
    constexpr uint32_t Wp = get_compile_time_arg_val(11);
    constexpr bool gamma_resident = get_compile_time_arg_val(12) != 0;
    constexpr auto output_args = TensorAccessorArgs<13>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t w0 = get_arg_val<uint32_t>(4);
    const uint32_t V = get_arg_val<uint32_t>(5);

    // ---- γ supply ----
    // Read row 0 of γ tiles [c*C, min(c*C + C, V)) of the window into the C pages at l1_chunk (page j = tile c*C + j);
    // pages past V are don't-care (they only scale x pad tiles, whose outputs are never written).
    [[maybe_unused]] auto read_gamma_chunk = [&](uint32_t l1_chunk, uint32_t c) {
        constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * es_g;
        constexpr uint32_t half_row_bytes = FACE_DIM * es_g;
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
        const uint32_t first = c * C;
        const uint32_t last = (first + C) < V ? first + C : V;
        for (uint32_t t = first; t < last; ++t) {
            const uint32_t page = l1_chunk + (t - first) * T_g;
            if constexpr (gamma_is_tile) {
                noc_async_read(gamma_accessor.get_noc_addr(w0 + t), page, face_bytes + half_row_bytes);
            } else {
                noc_async_read(gamma_accessor.get_noc_addr(w0 + t), page, 2 * half_row_bytes);
            }
        }
        noc_async_read_barrier();
        if constexpr (!gamma_is_tile) {
            // values 16..31 of each stick: face 0 row 1 -> face 1 row 0
            for (uint32_t t = first; t < last; ++t) {
                const uint32_t page = l1_chunk + (t - first) * T_g;
                volatile tt_l1_ptr uint32_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + half_row_bytes);
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + face_bytes);
                for (uint32_t i = 0; i < half_row_bytes / 4; ++i) {
                    dst[i] = src[i];
                }
            }
        }
    };
    // Streaming: one C-page chunk per (block, chunk) step into the STREAM_DEPTH * C ring.
    [[maybe_unused]] auto push_gamma_chunk = [&](uint32_t c) {
        cb_reserve_back(cb_gamma_tiles, C);
        read_gamma_chunk(get_write_ptr(cb_gamma_tiles), c);
        cb_push_back(cb_gamma_tiles, C);
    };

    const uint32_t num_blocks = (rows + R - 1) / R;
    if constexpr (has_gamma) {
        if constexpr (gamma_resident) {
            // One reservation of the whole window, pushed chunk by chunk (compute waits cumulatively, keeps it).
            cb_reserve_back(cb_gamma_tiles, Wp);
            const uint32_t base = get_write_ptr(cb_gamma_tiles);
            for (uint32_t c = 0; c < num_w_chunks; ++c) {
                read_gamma_chunk(base + c * C * T_g, c);
                cb_push_back(cb_gamma_tiles, C);
            }
        } else if (num_blocks > 0) {
            push_gamma_chunk(0);
        }
    }

    // ---- store_wave (+ streaming γ: step s+1's chunk is pushed BEFORE output step s is drained, so compute always
    // finds the next chunk ready. Deadlock-free: the reserve for step s+1 needs step s-1 popped, which compute did at
    // the end of scale_by_gamma(s-1), before it produced output step s-1, already drained here.) ----
    constexpr bool gamma_streams = has_gamma && !gamma_resident;
    if constexpr (!sharded || gamma_streams) {
        [[maybe_unused]] const auto out_accessor = TensorAccessor(output_args, out_addr, T_out);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rem = rows - b * R;
            const uint32_t rb = rem < R ? rem : R;
            for (uint32_t c = 0; c < num_w_chunks; ++c) {
                if constexpr (gamma_streams) {
                    const bool tail = (c + 1 == num_w_chunks);
                    if (!(tail && b + 1 == num_blocks)) {
                        push_gamma_chunk(tail ? 0 : c + 1);
                    }
                }
                if constexpr (sharded) {
                    continue;  // compute packs straight into the output shard
                }
                const uint32_t first = c * C;
                const uint32_t left = V > first ? V - first : 0;
                const uint32_t valid = left < C ? left : C;
                for (uint32_t r = 0; r < rb; ++r) {
                    const uint32_t tile_row = start_tile_row + b * R + r;
                    cb_wait_front(cb_output, C);
                    const uint32_t l1 = get_read_ptr(cb_output);
                    const uint32_t page0 = tile_row * Wt + w0 + first;
                    for (uint32_t j = 0; j < valid; ++j) {
                        noc_async_write_tile(page0 + j, out_accessor, l1 + j * T_out);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_output, C);
                }
            }
        }
        noc_async_write_barrier();
    }
}
