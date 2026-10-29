// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm writer (BRISC, NoC1) on every worker core.
//   γ supply (once per kernel, before any output): the slice's V row-0 γ tiles into cb_gamma_tiles —
//     TILE γ:          one DRAM read of the row-0 prefix of each tile (face 0 row 0 .. face 1 row 0; base
//                      GAMMA_DIRECT_READ = PREFIX)
//     ROW_MAJOR sticks: stick t (32 γ values of tile column t) read to the page start (fills face 0 rows 0-1), then
//                      elements 16..31 moved L1 -> L1 to face 1 row 0 (BroadcastDim::Row reads row 0 only)
//     streamed (L1 fallback GAMMA_STREAM): one C-tile chunk per (wave with rows, chunk), one item ahead of the stores
//   store_wave: per wave, per C-chunk, per tile-row: the chunk's valid tiles -> the output TensorAccessor (interleaved
//     DRAM / L1, or a sharded output with a placement other than the input's: resharding — its pages live on their
//     shard cores); output with the input's own shard placement: the output CB is backed on the output shard
//     (compute packs in place) — nothing to write.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Measurement-only stage zones (host DRMS_PERF_ZONES=1 -> define DRMS_PERF_ZONES; off in production).
#if defined(DRMS_PERF_ZONES) && DRMS_PERF_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif

namespace {
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_output = 16;
constexpr uint32_t FACE_DIM = 16;
}  // namespace

void kernel_main() {
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(0) != 0;
    constexpr bool OUT_SHARDED = get_compile_time_arg_val(1) != 0;  // output shard-backed: nothing to store
    constexpr uint32_t WS = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t NCH = get_compile_time_arg_val(4);
    constexpr uint32_t NW = get_compile_time_arg_val(5);
    constexpr uint32_t Q_W = get_compile_time_arg_val(6);
    constexpr uint32_t WT = get_compile_time_arg_val(7);
    constexpr uint32_t T_OUT = get_compile_time_arg_val(8);
    constexpr bool GAMMA_IS_TILE = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t GAMMA_ES = get_compile_time_arg_val(10);
    constexpr uint32_t T_G = get_compile_time_arg_val(11);
    constexpr uint32_t WP = get_compile_time_arg_val(12);
    // γ streams one C-tile chunk per (wave with rows, chunk) instead of the resident slice (L1 fallback)
    constexpr bool GAMMA_STREAM = get_compile_time_arg_val(13) != 0;
    constexpr auto output_args = TensorAccessorArgs<14>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t w0 = get_arg_val<uint32_t>(4);
    const uint32_t V = get_arg_val<uint32_t>(5);

    auto n_wave = [&](uint32_t w) -> uint32_t {
        const uint32_t r0 = w * Q_W;
        return r0 >= rows ? 0 : ((rows - r0) < Q_W ? rows - r0 : Q_W);
    };

    // γ tiles [t0, t0 + nt) of the slice -> pages 0.. of the cb_gamma_tiles write window (row 0 is all any consumer
    // reads: BroadcastDim::Row)
    [[maybe_unused]] auto read_gamma = [&](uint32_t t0, uint32_t nt) {
        if constexpr (HAS_GAMMA) {
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
            const uint32_t base = get_write_ptr(cb_gamma_tiles);
            if constexpr (GAMMA_IS_TILE) {
                constexpr uint32_t prefix = FACE_DIM * FACE_DIM * GAMMA_ES + FACE_DIM * GAMMA_ES;
                for (uint32_t t = 0; t < nt; ++t) {
                    noc_async_read(gamma_accessor.get_noc_addr(w0 + t0 + t), base + t * T_G, prefix);
                }
                noc_async_read_barrier();
            } else {
                constexpr uint32_t stick = 2 * FACE_DIM * GAMMA_ES;
                for (uint32_t t = 0; t < nt; ++t) {
                    noc_async_read(gamma_accessor.get_noc_addr(w0 + t0 + t), base + t * T_G, stick);
                }
                noc_async_read_barrier();
                for (uint32_t t = 0; t < nt; ++t) {
                    const uint32_t page = base + t * T_G;
                    noc_async_write(
                        page + FACE_DIM * GAMMA_ES,
                        get_noc_addr(page + FACE_DIM * FACE_DIM * GAMMA_ES),
                        FACE_DIM * GAMMA_ES);
                }
                noc_async_write_barrier();
            }
        }
    };
    auto chunk_width = [](uint32_t c) -> uint32_t { return (c + 1 == NCH) ? WS - c * C : C; };
    auto chunk_valid = [&](uint32_t c) -> uint32_t {
        const uint32_t t0 = c * C;
        const uint32_t cw = chunk_width(c);
        return t0 >= V ? 0 : ((V - t0) < cw ? V - t0 : cw);
    };

    // ---------------- γ supply ----------------
    // resident (once per kernel, before any output): the slice's V row-0 γ tiles into cb_gamma_tiles
    if constexpr (HAS_GAMMA && !GAMMA_STREAM) {
        DRMS_ZONE("wr_gamma");
        cb_reserve_back(cb_gamma_tiles, WP);
        read_gamma(0, V);
        cb_push_back(cb_gamma_tiles, WP);
    }
    // streamed: one chunk per (wave with rows, chunk), in the compute's consumption order
    uint32_t gw = 0, gc = 0;  // next γ chunk item
    [[maybe_unused]] auto push_gamma_chunk = [&]() -> bool {
        if constexpr (HAS_GAMMA && GAMMA_STREAM) {
            while (gw < NW && n_wave(gw) == 0) {
                ++gw;
            }
            if (gw >= NW) {
                return false;
            }
            DRMS_ZONE("wr_gamma");
            cb_reserve_back(cb_gamma_tiles, C);
            read_gamma(gc * C, chunk_valid(gc));
            cb_push_back(cb_gamma_tiles, C);
            if (++gc == NCH) {
                gc = 0;
                ++gw;
            }
            return true;
        }
        return false;
    };

    // ---------------- store_wave ----------------
    if constexpr (OUT_SHARDED) {
        // output shard-backed (compute packs in place): only the streamed γ remains to supply
        while (push_gamma_chunk()) {
        }
    } else {
        // interleaved or resharded output (TensorAccessor: a sharded output's pages live on their shard cores)
        const auto output_accessor = TensorAccessor(output_args, out_addr);
        push_gamma_chunk();  // one chunk ahead: chunk i+1's γ read overlaps chunk i's stores
        for (uint32_t w = 0; w < NW; ++w) {
            const uint32_t r0 = w * Q_W;
            const uint32_t n = n_wave(w);
            for (uint32_t c = 0; c < NCH && n > 0; ++c) {
                push_gamma_chunk();
                const uint32_t t0 = c * C;
                const uint32_t cw = chunk_width(c);  // a shard-backed slice's last chunk is narrower
                const uint32_t nv = chunk_valid(c);
                for (uint32_t r = 0; r < n; ++r) {
                    {
                        DRMS_ZONE("wr_wait_out");
                        cb_wait_front(cb_output, cw);
                    }
                    DRMS_ZONE("wr_store");
                    const uint32_t l1 = get_read_ptr(cb_output);
                    const uint32_t page0 = (start_row + r0 + r) * WT + w0 + t0;
                    for (uint32_t i = 0; i < nv; ++i) {
                        noc_async_write_tile(page0 + i, output_accessor, l1 + i * T_OUT);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_output, cw);
                }
            }
        }
        noc_async_write_barrier();
    }
}
