// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// argmax_rvv_tile_compute.cpp — TILE-layout last-dim argmax+maxval on the pack
// RISC's RVV (Zve32f) unit. Blackhole only; selected internally by
// ttnn.argmax (ArgMaxPath::Rvv).
//
// The scalar-reader TILE-input argmax has no compute kernel: the whole reduction
// is a scalar C++ loop on a single dataflow RISC, with the NOC read and the
// scan of each tile serialized against each other. Here the reader streams
// tiles into a double-buffered CB, the scan of one chunk overlaps the staging
// of the next, and this kernel scans with 16-lane bf16 vector ops:
//
//   - TRISC0/1 (unpack/math) are complete no-ops: no Tensix instruction is
//     ever issued, so there is no LReg/MOP/ADDR_MOD state to conflict with.
//   - TRISC2 consumes the input CB and produces the per-row (index, maxval)
//     results via the raw stream-register CB protocol (the counters are
//     NOC-overlay scratch registers, MMIO-visible from any RISC; this kernel
//     is the sole acker of the input CB and sole producer of the output CBs).
//
// Algorithm (per outer index, per 32-row tile-row, chunk-streamed):
//   Tiles arrive in chunks of CHUNK_PAGES. Rows are processed in pairs: the
//   two 16-element face rows of a row-pair are contiguous in a tile face, so
//   one e16m4 (vl=32) load covers rows {2g, 2g+1} x cols 0..15 of one face.
//   Pass A per chunk accumulates a lane-wise unsigned max of x ^ 0x8000 over
//   the chunk's tiles (2 loads + 2 xor + 2 vmaxu = 6 vector instrs per tile
//   per row-pair). At the chunk boundary each valid row's 32 accumulator
//   lanes are reduced; if the chunk's row-max beats the running max, the
//   still-resident chunk is re-scanned (vmseq + vfirst on the raw bits) for
//   the first occurrence.
//
// Semantics — exactly ttnn.argmax's bfloat16_greater + std::min tie-break:
//   bfloat16_greater is a pure sign-magnitude bit-pattern total order. The
//   xor trick (t = x ^ 0x8000) makes unsigned lane max agree with that order
//   whenever the lane set contains any sign-0 pattern; a chunk-row whose max
//   transformed value is < 0x8000 was all-negative, and takes an exact minu
//   fix-up sweep (both-negative order is reversed). Comparisons across chunks
//   happen in the fully monotone domain m = x ^ ((x >> 15 arith) | 0x8000);
//   the running max is seeded with m(0xFF80) — the scalar readers' -inf init
//   — so even the all-negative-NaN-row corner matches them bit-for-bit.
//   Strictly-greater updates + first-match re-scan preserve the smallest-
//   index tie-break across lanes, faces, tiles, and chunks.
//
// Multicore: this kernel only ever scans its own slice of the tile-row —
//   w_count tiles, a runtime arg — so the indices it emits are local to that
//   slice (element 0 of the slice is index 0). The reader adds the slice's
//   w_start * 32 offset before any cross-core compare, and merges the
//   per-core candidates in the same bit-pattern total order; see
//   reader_argmax_rvv_tile.cpp. Nothing about the scan itself changes: a core
//   with the whole row (num_cores == 1) sees exactly the single-core case.
//
// Register budget (hard-earned): e16m4 dual-stream uses 16 of 32 vregs.
// e16m8 dual-stream needs all 32 and GCC spills a multi-KB stack frame
// against the pack RISC's ~256B stack — instant overflow hang. Keep helpers
// noinline and check any future change's prologue for vlenb-scaled sub sp.
// =============================================================================

#include <cstdint>
#include "api/compute/common.h"
#include "hostdevcommon/kernel_structs.h"

#ifdef TRISC_PACK
#include <riscv_vector.h>
#include "internal/tt-1xx/risc_common.h"  // invalidate_l1_cache()

namespace {

// ---- raw stream-register CB protocol (sole acker / sole producer) ----------
inline uint16_t amx_tiles_received(uint32_t cb) {
    return (uint16_t)reg_read((uint32_t)(uintptr_t)get_cb_tiles_received_ptr((int)cb));
}

inline void amx_in_wait(uint32_t cb, uint16_t my_acked, uint16_t want) {
    while ((uint16_t)(amx_tiles_received(cb) - my_acked) < want) {
    }
}

inline void amx_in_pop(uint32_t cb, uint16_t& my_acked, uint16_t n) {
    my_acked += n;
    get_cb_tiles_acked_ptr((int)cb)[0] = my_acked;
}

inline void amx_out_reserve(uint32_t cb, uint16_t my_received, uint16_t n) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    while ((uint16_t)((uint16_t)i.fifo_num_pages -
                      (uint16_t)(my_received -
                                 (uint16_t)reg_read((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr((int)cb)))) < n) {
    }
}

// Publish `n` pages of CB `cb` after the caller has written `bytes_written`
// bytes starting at the (16B-aligned) page base `page_addr`. Stores from this
// RISC drain to L1 in program order, so reading back the aligned 4-byte word
// that contains the last written byte — offset (bytes_written - 1) & ~3 —
// guarantees every store into the page is L1-visible before the received
// counter moves. Rounding down is required, not optional: the read-back has to
// be the aligned uint32_t load whose span covers the final byte. So
// bytes_written == 2 fences word 0, and word 0 is the word holding that last
// written half-word — this is not an "early" fence.
inline void amx_out_push(uint32_t cb, uint16_t& my_received, uint32_t page_addr, uint32_t bytes_written, uint16_t n) {
    asm volatile("fence" ::: "memory");
    (void)*(volatile uint32_t*)(page_addr + ((bytes_written - 1u) & ~3u));
    my_received += n;
    get_cb_tiles_received_ptr((int)cb)[0] = my_received;
}

// Byte address of global page `t` of CB `cb`. Pack-side CB init leaves
// fifo_wr_ptr == base and this thread never llk-pushes, so base is stable.
inline uint32_t amx_page_addr(uint32_t cb, uint32_t t) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    const uint32_t slot = t % i.fifo_num_pages;
    return (i.fifo_wr_ptr + slot * i.fifo_page_size) << 4;  // 16B units
}

// Wrap-around iterator over consecutive CB pages (keeps the hot loops free of
// the integer divide amx_page_addr costs per call).
struct AmxPageIter {
    uint32_t addr;
    uint32_t base;
    uint32_t limit;
    uint32_t step;

    AmxPageIter(uint32_t cb, uint32_t t0) {
        LocalCBInterface& i = get_local_cb_interface(cb);
        base = i.fifo_wr_ptr << 4;
        step = i.fifo_page_size << 4;
        limit = base + i.fifo_num_pages * step;
        addr = amx_page_addr(cb, t0);
    }

    inline uint32_t next() {  // returns current page address, then advances
        const uint32_t cur = addr;
        addr += step;
        if (addr >= limit) {
            addr = base;
        }
        return cur;
    }
};

// Face-quadrant byte offsets of row-pair group `g` (rows 2g, 2g+1) within a
// 32x32 bf16 tile (faces stored 0,1,2,3; each face 16x16 row-major = 512B).
// Left face holds cols 0..15, right face (offset +512B) cols 16..31.
inline uint32_t amx_left_off(uint32_t g) { return ((g < 8) ? 0u : 1024u) + (g & 7u) * 64u; }

constexpr uint16_t kSignBit = 0x8000u;
// Monotone image of 0xFF80 (-inf) — the scalar scan's initial max value.
constexpr uint16_t kInitMono = 0x007Fu;

// Running per-row state for the current tile-row pass. Static => resides in
// local data memory, not on the (tiny) stack.
uint16_t s_running_mono[32];
uint32_t s_running_raw[32];
uint32_t s_running_idx[32];

// ---- pass A + extraction + (rare) fix-up + (rare) first-index re-scan ------
// Processes row-pair group `g` of the resident chunk [t0, t0+chunk) and
// updates the running (mono, raw, idx) state for its valid rows.
__attribute__((noinline)) void amx_process_chunk_group(
    uint32_t cb_in, uint32_t t0, uint32_t chunk, uint32_t g, uint32_t rows_in_group, uint32_t tiles_done) {
    const uint32_t left_off = amx_left_off(g);
    const size_t vl = __riscv_vsetvl_e16m4(32);

    // Pass A: lane-wise max of x ^ 0x8000 across the chunk's tiles.
    vuint16m4_t acc_l = __riscv_vmv_v_x_u16m4(0, vl);
    vuint16m4_t acc_r = __riscv_vmv_v_x_u16m4(0, vl);
    AmxPageIter it(cb_in, t0);
    for (uint32_t t = 0; t < chunk; t++) {
        const uint32_t base = it.next();
        vuint16m4_t a = __riscv_vle16_v_u16m4((const uint16_t*)(base + left_off), vl);
        vuint16m4_t b = __riscv_vle16_v_u16m4((const uint16_t*)(base + left_off + 512u), vl);
        a = __riscv_vxor_vx_u16m4(a, kSignBit, vl);
        b = __riscv_vxor_vx_u16m4(b, kSignBit, vl);
        acc_l = __riscv_vmaxu_vv_u16m4(acc_l, a, vl);
        acc_r = __riscv_vmaxu_vv_u16m4(acc_r, b, vl);
    }

    for (uint32_t rr = 0; rr < rows_in_group; rr++) {
        const uint32_t row = 2 * g + rr;
        const uint32_t row_off = left_off + rr * 32u;  // 16 bf16 = 32B per face row

        // Reduce this row's 16 lanes of each accumulator (lanes 16*rr..).
        vuint16m4_t sl = rr ? __riscv_vslidedown_vx_u16m4(acc_l, 16, vl) : acc_l;
        vuint16m4_t sr = rr ? __riscv_vslidedown_vx_u16m4(acc_r, 16, vl) : acc_r;
        const vuint16m1_t z = __riscv_vmv_s_x_u16m1(0, 1);
        uint16_t t_row = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m4_u16m1(sl, z, 16));
        const uint16_t t_r = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m4_u16m1(sr, z, 16));
        if (t_r > t_row) {
            t_row = t_r;
        }

        uint16_t mono;
        uint16_t raw;
        if (t_row >= kSignBit) {
            // Some sign-0 pattern exists: the xor-domain max is the winner.
            mono = t_row;
            raw = (uint16_t)(t_row ^ kSignBit);
        } else {
            // All-negative chunk-row: both-negative order is reversed, take
            // the exact unsigned min over the (still resident) chunk.
            // NOTE: 16 lanes of e16 need m2 — VLEN=128 caps e16m1 at vl=8.
            const size_t vl1 = __riscv_vsetvl_e16m2(16);
            vuint16m2_t mn = __riscv_vmv_v_x_u16m2(0xFFFFu, vl1);
            AmxPageIter fit(cb_in, t0);
            for (uint32_t t = 0; t < chunk; t++) {
                const uint32_t base = fit.next();
                vuint16m2_t a = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off), vl1);
                vuint16m2_t b = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off + 512u), vl1);
                a = __riscv_vxor_vx_u16m2(a, kSignBit, vl1);
                b = __riscv_vxor_vx_u16m2(b, kSignBit, vl1);
                mn = __riscv_vminu_vv_u16m2(mn, a, vl1);
                mn = __riscv_vminu_vv_u16m2(mn, b, vl1);
            }
            const uint16_t t_min =
                __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m2_u16m1(mn, __riscv_vmv_s_x_u16m1(0xFFFFu, 1), vl1));
            mono = (uint16_t)(t_min ^ 0x7FFFu);
            raw = (uint16_t)(t_min ^ kSignBit);
        }

        // Strictly-greater update in the monotone domain keeps the earliest
        // occurrence across chunks; first-match re-scan keeps it in-chunk.
        if (mono > s_running_mono[row]) {
            const size_t vl1 = __riscv_vsetvl_e16m2(16);
            uint32_t idx = 0;
            AmxPageIter sit(cb_in, t0);
            for (uint32_t t = 0; t < chunk; t++) {
                const uint32_t base = sit.next();
                const vuint16m2_t a = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off), vl1);
                const long fa = __riscv_vfirst_m_b8(__riscv_vmseq_vx_u16m2_b8(a, raw, vl1), vl1);
                if (fa >= 0) {
                    idx = (tiles_done + t) * 32u + (uint32_t)fa;
                    break;
                }
                const vuint16m2_t b = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off + 512u), vl1);
                const long fb = __riscv_vfirst_m_b8(__riscv_vmseq_vx_u16m2_b8(b, raw, vl1), vl1);
                if (fb >= 0) {
                    idx = (tiles_done + t) * 32u + 16u + (uint32_t)fb;
                    break;
                }
            }
            s_running_mono[row] = mono;
            s_running_raw[row] = raw;
            s_running_idx[row] = idx;
        }
    }
}

}  // namespace
#endif  // TRISC_PACK

void kernel_main() {
#ifndef TRISC_PACK
    // unpack + math threads: intentionally empty — no Tensix instructions.
#else
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_idx = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_val = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_pages = get_compile_time_arg_val(3);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t logical_height = get_compile_time_arg_val(5);
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(6);

    // This core's slice of the reduction dim's tiles. Runtime, not compile
    // time: the leading cores carry one extra tile when w_tiles does not
    // divide evenly, and every core must share one compiled kernel.
    const uint32_t w_count = get_arg_val<uint32_t>(0);

    uint16_t acked_in = 0;  // local mirror of cb_in acked (sole acker)
    uint16_t recv_idx = 0;  // local mirror of cb_out_idx received (sole producer)
    uint16_t recv_val = 0;  // local mirror of cb_out_val received (sole producer)
    uint32_t t_global = 0;  // global input page counter (matches the reader)

    for (uint32_t outer = 0; outer < outer_dim_units; outer++) {
        for (uint32_t i = 0; i < h_tiles; i++) {
            const uint32_t row_base = i * 32u;
            const uint32_t units = (logical_height - row_base < 32u) ? (logical_height - row_base) : 32u;
            const uint32_t n_groups = (units + 1) / 2;

            for (uint32_t r = 0; r < units; r++) {
                s_running_mono[r] = kInitMono;
                s_running_raw[r] = 0xFF80u;  // -inf: the scalar readers' init value
                s_running_idx[r] = 0;
            }

            uint32_t tiles_done = 0;
            while (tiles_done < w_count) {
                const uint32_t chunk = (w_count - tiles_done < chunk_pages) ? (w_count - tiles_done) : chunk_pages;
                amx_in_wait(cb_in, acked_in, (uint16_t)chunk);
                invalidate_l1_cache();
                for (uint32_t g = 0; g < n_groups; g++) {
                    const uint32_t rows_in_group = (units - 2 * g < 2u) ? (units - 2 * g) : 2u;
                    amx_process_chunk_group(cb_in, t_global, chunk, g, rows_in_group, tiles_done);
                }
                amx_in_pop(cb_in, acked_in, (uint16_t)chunk);
                t_global += chunk;
                tiles_done += chunk;
            }

            // Emit this tile-row pass's results: one page of indices (u32)
            // and one page of max values (bf16 raw bits).
            amx_out_reserve(cb_out_idx, recv_idx, 1);
            amx_out_reserve(cb_out_val, recv_val, 1);
            const uint32_t ipage = amx_page_addr(cb_out_idx, recv_idx);
            const uint32_t vpage = amx_page_addr(cb_out_val, recv_val);
            volatile uint32_t* ip = (volatile uint32_t*)ipage;
            volatile uint16_t* vp = (volatile uint16_t*)vpage;
            for (uint32_t r = 0; r < units; r++) {
                ip[r] = s_running_idx[r];
                vp[r] = (uint16_t)s_running_raw[r];
            }
            // val page: `units` contiguous bf16 raw-bit half-words (2B each);
            // idx page: `units` contiguous u32 indices (4B each). The helper
            // fences on the aligned word containing the last written byte.
            amx_out_push(cb_out_val, recv_val, vpage, 2u * units, 1);
            amx_out_push(cb_out_idx, recv_idx, ipage, 4u * units, 1);
        }
    }
#endif  // TRISC_PACK
}
