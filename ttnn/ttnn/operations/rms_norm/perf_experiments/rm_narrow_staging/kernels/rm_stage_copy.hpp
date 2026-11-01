// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rm_narrow_staging bench — RISC-V L1 load/store copy of one RM stick piece (no NoC transaction).
//
// Bypassed: noc_async_read / noc_async_write of a local (own-L1) stick piece. Mechanism: for a narrow piece the
// cost of a NoC transaction is its ISSUE (command-buffer wait + ~6 register writes + counter bump, tens of RISC-V
// cycles) plus the barrier/flush, not the bytes; a 16 B piece is 4 word loads + 4 word stores on the RISC-V.
// No helper exists for an L1 -> L1 pitch-changing copy (fill_l1_range covers only the constant fill).
//
// Variant selection (bench only, via pd.PERF_ABLATE named defines):
//   RMS_CPU_MAX_<n>  pieces of <= n bytes use the RISC-V copy (default 0 = never — the op's current NoC path)
//   RMS_CONTIG       shard row pitch == staging pitch and full piece: one NoC transaction per tile-row
//   RMS_STATEFUL     per-stick local NoC transactions via the one-packet set_state / with_state pair: the NoC
//                    coordinates, length and command config are programmed once per tile-row, each stick then
//                    only writes its two L1 addresses and fires (pieces are <= C*32*4 B <= NOC_MAX_BURST_SIZE)
//   RMS_PREZERO      single-chunk shards: the staging tail is zeroed once per cb_x_rm slot, not per piece
#pragma once

#include <cstdint>

#if defined(RMS_CPU_MAX_4096)
constexpr uint32_t cpu_copy_max_bytes = 4096;
#elif defined(RMS_CPU_MAX_1024)
constexpr uint32_t cpu_copy_max_bytes = 1024;
#elif defined(RMS_CPU_MAX_512)
constexpr uint32_t cpu_copy_max_bytes = 512;
#elif defined(RMS_CPU_MAX_256)
constexpr uint32_t cpu_copy_max_bytes = 256;
#elif defined(RMS_CPU_MAX_128)
constexpr uint32_t cpu_copy_max_bytes = 128;
#elif defined(RMS_CPU_MAX_64)
constexpr uint32_t cpu_copy_max_bytes = 64;
#elif defined(RMS_CPU_MAX_32)
constexpr uint32_t cpu_copy_max_bytes = 32;
#elif defined(RMS_CPU_MAX_16)
constexpr uint32_t cpu_copy_max_bytes = 16;
#else
constexpr uint32_t cpu_copy_max_bytes = 0;
#endif
#ifdef RMS_CONTIG
constexpr bool rm_contig = true;
#else
constexpr bool rm_contig = false;
#endif
#ifdef RMS_STATEFUL
constexpr bool rm_stateful = true;
#else
constexpr bool rm_stateful = false;
#endif
#ifdef RMS_PREZERO
constexpr bool rm_prezero = true;
#else
constexpr bool rm_prezero = false;
#endif

// Copy `bytes` (multiple of 2; src/dst 4 B aligned) from src to dst with word loads/stores. Four loads are issued
// ahead of their stores so the in-order RISC-V overlaps the L1 load latency.
FORCE_INLINE void l1_copy_piece(uint32_t dst, uint32_t src, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    const uint32_t words = bytes >> 2;
    uint32_t w = 0;
    for (; w + 4 <= words; w += 4) {
        const uint32_t a = s[w], b = s[w + 1], c = s[w + 2], e = s[w + 3];
        d[w] = a;
        d[w + 1] = b;
        d[w + 2] = c;
        d[w + 3] = e;
    }
    for (; w < words; ++w) {
        d[w] = s[w];
    }
    if (bytes & 2) {
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst)[words * 2] =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src)[words * 2];
    }
}

// Zero [from, to) (both multiples of 2, `to` 4 B aligned).
FORCE_INLINE void l1_zero_range(uint32_t from, uint32_t to) {
    if (from & 2) {
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(from) = 0;
        from += 2;
    }
    for (volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(from);
         p < reinterpret_cast<volatile tt_l1_ptr uint32_t*>(to);
         ++p) {
        *p = 0;
    }
}
