// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize NoC / indexing primitives shared by the reader and the writer
// (Refinement 6 — the run-closing completeness audit).
//
// Two catalog levers live here, both built so their OFF arm is byte-identical
// to the code that shipped before them:
//
//   master.md B13 — `set_state` / `with_state` STATEFUL transfers.
//     A NoC transaction is programmed by writing a handful of command-buffer
//     registers. `*_set_state` writes the ENDPOINT ones (the destination /
//     source COORDINATE, and the length for the one-packet form); `*_with_state`
//     then patches only the local addresses and issues. On Wormhole
//     (`noc_nonblocking_api.h`) a plain `ncrisc_noc_fast_read` writes FIVE
//     registers (RET_ADDR_LO, TARG_ADDR_LO, TARG_ADDR_COORDINATE, AT_LEN_BE,
//     CMD_CTRL); the stateful pair writes ONE (COORD) + FOUR. So the lever pays
//     exactly one register write per transfer, and ONLY while consecutive
//     transfers share an endpoint — which is why the state below is cached on
//     the coordinate field and reprogrammed the moment it changes. On a path
//     whose transfers rotate endpoints (every interleaved tensor: a tile-row is
//     TILE_H consecutive pages, and page `p` lives in bank `p % num_banks`) the
//     arm degrades to one extra `noc_cmd_buf_ready` poll, never to wrong data.
//
//   master.md D21 — per-core block indexing precomputed HOST-side.
//     A W_BLOCKS core owns a contiguous range of the global, W-chunk-major
//     block index, so block `b` is `(row = b % nt_h, chunk = b / nt_h)`. The
//     kernel can either recompute that div/mod every block (OFF) or take the
//     range's origin from the host and step it (ON). Same index either way.

#pragma once

#include <cstdint>

namespace tilize_kernels {

// ---------------------------------------------------------------------------
// master.md B13 — stateful reads
// ---------------------------------------------------------------------------
// `enabled == false` compiles to exactly the `noc_async_read` call site the
// caller had before, so the OFF arm is the prior kernel byte for byte.
//
// `packet_bytes` COMPOSES B13 with master.md B6 instead of replacing it. When
// every transfer is the same size and fits one packet, the one-packet stateful
// pair puts the LENGTH in the state too (set_state writes COORD + AT_LEN;
// with_state writes only RET_ADDR_LO, TARG_ADDR_LO, CMD_CTRL). Pass 0 for the
// any-length form, which a variable-length gather needs. Measuring the
// any-length arm against a caller that was already on B6's one-packet path
// would price B13 against B6 rather than against the barrier it removes —
// which is exactly what the first Refinement-6 measurement did, and why this
// parameter exists.
template <bool enabled, uint32_t packet_bytes = 0>
struct StatefulRead {
    uint32_t coord = 0xFFFFFFFFu;  // no endpoint programmed yet

    FORCE_INLINE void read(uint64_t src_noc_addr, uint32_t l1_addr, uint32_t n_bytes) {
        if constexpr (!enabled) {
            noc_async_read(src_noc_addr, l1_addr, n_bytes);
        } else {
            const uint32_t c = (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT);
            if constexpr (packet_bytes) {
                if (c != coord) {
                    noc_async_read_one_packet_set_state(src_noc_addr, packet_bytes);
                    coord = c;
                }
                noc_async_read_one_packet_with_state((uint32_t)src_noc_addr, l1_addr);
            } else {
                if (c != coord) {
                    noc_async_read_set_state(src_noc_addr);
                    coord = c;
                }
                noc_async_read_with_state((uint32_t)src_noc_addr, l1_addr, n_bytes);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// master.md B13 — stateful writes (the reader lever's twin)
// ---------------------------------------------------------------------------
// The public API has no ANY-LENGTH stateful write: only
// `noc_async_write_one_packet_{set,with}_state`, whose contract is a transfer
// of at most NOC_MAX_BURST_SIZE. `xfer_bytes` is therefore a compile-time
// constant (the output TILE page) and the host only turns the lever on when
// that page fits one packet — which is a TILE-HEIGHT question, not a dtype one
// (a 32-row bf16 tile page is 2048 B; an 8-row one is 512 B). That bound is the
// transaction-size sweep this lever has to be priced across.
template <bool enabled, uint32_t xfer_bytes>
struct StatefulWrite {
    static_assert(!enabled || xfer_bytes <= NOC_MAX_BURST_SIZE, "stateful write is the one-packet form only");
    uint32_t coord = 0xFFFFFFFFu;

    FORCE_INLINE void write(uint32_t l1_addr, uint64_t dst_noc_addr) {
        if constexpr (enabled) {
            const uint32_t c = (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT);
            if (c != coord) {
                noc_async_write_one_packet_set_state(dst_noc_addr, xfer_bytes);
                coord = c;
            }
            noc_async_write_one_packet_with_state(l1_addr, (uint32_t)dst_noc_addr);
        } else {
            noc_async_write(l1_addr, dst_noc_addr, xfer_bytes);
        }
    }
};

// ---------------------------------------------------------------------------
// master.md D21 — the W_BLOCKS block index
// ---------------------------------------------------------------------------
// THE single source for "block i of this core -> (tile-row, W chunk)". Both
// arms produce the same pair; they differ only in whether the div/mod is paid
// per block in the kernel or once on the host.
// Usage is always the same three calls, so the two arms cost the caller
// nothing structural:
//
//     BlockIndex<precomp> idx; idx.init(start_block, row0, wc0, nt_h);
//     for (i ...) { idx.seek(i); /* use idx.row / idx.wc */ idx.advance(); }
//
// With `precomputed == false`, `seek()` recomputes the div/mod exactly as the
// pre-Refinement-6 kernels did and `advance()` compiles away — so the OFF arm
// is the prior code byte for byte. With `precomputed == true` it is the other
// way round: the host did the one division, and the loop only steps.
//
// `nt_h` is a TEMPLATE parameter, not a member, and that is load-bearing: it is
// a compile-time arg in both kernels, and the wide/short regime has `nt_h == 1`,
// where the compiler folds `b % 1` to 0 and `b / 1` to `b` outright. Carrying it
// as a member would turn a folded-away divide into a real one on exactly the
// shape that has the most blocks per core — a silent pessimization of the OFF
// arm, i.e. of the shipped kernel.
template <bool precomputed, uint32_t nt_h>
struct BlockIndex {
    uint32_t row = 0;  // tile-row of the current block
    uint32_t wc = 0;   // W chunk of the current block
    uint32_t start_block_ = 0;

    // `row0` / `wc0` are the host's decomposition of `start_block`.
    FORCE_INLINE void init(uint32_t start_block, uint32_t row0, uint32_t wc0) {
        start_block_ = start_block;
        if constexpr (precomputed) {
            row = row0;
            wc = wc0;
        }
    }

    FORCE_INLINE void seek(uint32_t i) {
        if constexpr (!precomputed) {
            const uint32_t b = start_block_ + i;
            row = b % nt_h;
            wc = b / nt_h;
        }
    }

    // Blocks are W-chunk-major, so the tile-row is the fast axis and the chunk
    // carries.
    FORCE_INLINE void advance() {
        if constexpr (precomputed) {
            ++row;
            if (row == nt_h) {
                row = 0;
                ++wc;
            }
        }
    }
};

}  // namespace tilize_kernels
