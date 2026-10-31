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
//   write flush per quantum, through the output TensorAccessor (interleaved, or a shard placed differently from the
//   input: a non-local write). Native (output shard == input shard): cb_output IS the output shard, compute packs
//   into it; nothing to write.
// Output on both NoCs (host OUT_NOC_SPLIT, DRAM-interleaved output >= OUT_NOC_SPLIT_MIN_BYTES): a tile whose DRAM
//   bank (page % NUM_DRAM_BANKS) is in this core's NOC0 set (default: the odd banks; the set may depend on the core's
//   physical x against a DRAM column, the "direction" mode) is written by the READER on NOC0 instead of here on NOC1,
//   so the burst (DRAM-write-bound) is injected on both NoCs. This writer stays cb_output's ONLY consumer: per quantum
//   it posts the quantum's L1 address + count on two local semaphore words, writes its NOC1 tiles, and pops only once
//   the reader reported its NOC0 tiles flushed (one quantum outstanding).
// Raw NoC / CB calls: no kernel_lib helper reads a γ row-0 prefix or relays a γ stick into a tile's row 0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// Measurement-only per-stage zones (host DRMS_ZONES=1 -> define; never set in production).
#ifdef DRMS_ZONES
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif

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
    constexpr bool sharded = get_compile_time_arg_val(7) != 0;  // output native (zero-copy shard)
    constexpr uint32_t T_out = get_compile_time_arg_val(8);
    constexpr uint32_t T_g = get_compile_time_arg_val(9);
    constexpr uint32_t es_g = get_compile_time_arg_val(10);
    constexpr uint32_t Wp = get_compile_time_arg_val(11);
    constexpr bool gamma_resident = get_compile_time_arg_val(12) != 0;
    // Output NoC split (0 / 0: off, every tile on this NoC). The same CT values reach the reader.
    constexpr uint32_t out_split_x = get_compile_time_arg_val(13);
    constexpr uint32_t noc0_banks_west = get_compile_time_arg_val(14);
    constexpr uint32_t noc0_banks_east = get_compile_time_arg_val(15);
    constexpr uint32_t sem_out_ready = get_compile_time_arg_val(16);  // writer -> reader: quanta posted
    constexpr uint32_t sem_out_done = get_compile_time_arg_val(17);   // reader -> writer: quanta flushed
    constexpr uint32_t sem_out_addr = get_compile_time_arg_val(18);   // L1 address of the posted quantum
    constexpr bool out_split = (noc0_banks_west != 0 || noc0_banks_east != 0) && !sharded;
    constexpr auto output_args = TensorAccessorArgs<19>();
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
            DRMS_ZONE("wr_gamma");
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
        // my_x[0]: this core's physical NoC0 x (the same value on the reader).
        [[maybe_unused]] const uint32_t noc0_banks = my_x[0] < out_split_x ? noc0_banks_west : noc0_banks_east;
        [[maybe_unused]] volatile tt_l1_ptr uint32_t* out_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_ready));
        [[maybe_unused]] volatile tt_l1_ptr uint32_t* out_done =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_done));
        [[maybe_unused]] volatile tt_l1_ptr uint32_t* out_l1 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_addr));
        [[maybe_unused]] uint32_t posted = 0;
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
                    DRMS_ZONE("wr_store");
                    cb_wait_front(cb_output, C);
                    const uint32_t l1 = get_read_ptr(cb_output);
                    const uint32_t page0 = tile_row * Wt + w0 + first;
                    if constexpr (out_split) {
                        *out_l1 = l1;
                        *out_ready = ++posted;  // after the address: the reader reads both once it sees the count
                    }
#ifndef DRMS_ABLATE_WRITE
                    for (uint32_t j = 0; j < valid; ++j) {
                        if constexpr (out_split) {
                            if ((noc0_banks >> ((page0 + j) % NUM_DRAM_BANKS)) & 1) {
                                continue;  // the reader's NOC0 share
                            }
                        }
                        noc_async_write_tile(page0 + j, out_accessor, l1 + j * T_out);
                    }
#endif
                    noc_async_writes_flushed();
                    if constexpr (out_split) {
                        do {
                            invalidate_l1_cache();
                        } while (*out_done < posted);
                    }
                    cb_pop_front(cb_output, C);
                }
            }
        }
        noc_async_write_barrier();
    }
}
