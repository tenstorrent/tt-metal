// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — the transport vocabulary shared by the READER (NoC0) and the WRITER (NoC1).
//
// These are the pieces the two dataflow kernels genuinely do the SAME way, and every one of them
// used to exist twice: the gate/up weight-chunk read, the W_down row read (three copies), the
// reduce-scatter gather leg, the peer-coordinate fetch (seven copies) and the semaphore idioms
// (~fifteen). A protocol expressed twice is a protocol that can disagree, and the two places that
// matter most here — the address arithmetic and the semaphore accounting — are silent when they do.
//
// Dataflow-only, so it is NOT include-free: the compute kernel must not see the dataflow API.
// Anything all three kernels need lives in `moe_fused_swiglu_common.hpp` instead.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "moe_fused_swiglu_bank_runs.hpp"
#include "moe_fused_swiglu_common.hpp"

namespace moe_fused_swiglu {

// Minimal operation-local multicast geometry.  kernel_lib's McastRect is not on
// origin/main; this is the only part of it this operation needs.  The hardware
// expects NOC0 to walk low-to-high and NOC1 high-to-low.
template <uint8_t NOC_ID = noc_index>
struct McastRect {
    struct Bounds {
        uint32_t sx, sy, ex, ey;
    };

    constexpr McastRect(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) :
        xlo_(x0 < x1 ? x0 : x1),
        xhi_(x0 < x1 ? x1 : x0),
        ylo_(y0 < y1 ? y0 : y1),
        yhi_(y0 < y1 ? y1 : y0),
        bounds_(NOC_ID == 1 ? Bounds{xhi_, yhi_, xlo_, ylo_} : Bounds{xlo_, ylo_, xhi_, yhi_}) {}

    constexpr const Bounds& bounds() const { return bounds_; }

    constexpr uint32_t area() const {
        uint32_t width = xhi_ - xlo_ + 1;
#if defined(ARCH_BLACKHOLE)
        width -= static_cast<uint32_t>(xlo_ <= 8 && 8 <= xhi_);
        width -= static_cast<uint32_t>(xlo_ <= 9 && 9 <= xhi_);
#endif
        return width * (yhi_ - ylo_ + 1);
    }

private:
    uint32_t xlo_, xhi_, ylo_, yhi_;
    Bounds bounds_;
};

// Semaphores. All but one are MONOTONE — never reset within a dispatch, compared against a running
// total, which is what makes them race-free across M-blocks. The exception is the h all-gather's
// per-slot VALID cells (SEM_H_RDY_BASE + s): a Flag, set and cleared each round.
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(id)));
}

FORCE_INLINE void sem_wait_min(uint32_t id, uint32_t target) { noc_semaphore_wait_min(sem_ptr(id), target); }

//: An intra-core publish: producer and consumer are two RISC-Vs on the same core sharing one L1,
//: and the word has exactly one writer, so a plain volatile store is the whole handshake.
FORCE_INLINE void sem_publish(uint32_t id, uint32_t value) { *sem_ptr(id) = value; }

// Peers. The grid COLUMN in virtual coordinates, KGROUPS (vx, vy) pairs in ROW order at `RT_PEERS`.
// Row r sits at index r on every core in the column, which is what lets "worker r owns tiles
// [r*a, (r+1)*a)" agree grid-wide with no host-side plan table.
struct Peer {
    uint32_t x, y;
};

FORCE_INLINE Peer peer_at(uint32_t rt_peers, uint32_t r) {
    return Peer{get_arg_val<uint32_t>(rt_peers + 2 * r + 0), get_arg_val<uint32_t>(rt_peers + 2 * r + 1)};
}

// Weight streams
//: One hidden-axis CHUNK of a gate/up block: this row's full K extent, `chunk_w` columns wide, at
//: CB row stride `chunk_w`. Reads are ISSUED only — the caller owns the barrier. `read == false`
//: reproduces the CB cycle with no DRAM traffic when weights are resident.
template <class Runs, class Acc>
FORCE_INLINE void read_weight_chunk(
    const Acc& acc,
    bool read,
    uint32_t chunk,
    uint32_t chunk_w,
    uint32_t kr_rows,
    uint32_t kstart,
    uint32_t hstart,
    uint32_t hn_cols,
    uint32_t hid_t,
    uint32_t l1_base,
    uint32_t tile_bytes) {
    const uint32_t chunk_col0 = chunk * chunk_w;
    uint32_t w = (chunk_col0 < hn_cols) ? (hn_cols - chunk_col0) : 0;
    if (w > chunk_w) {
        w = chunk_w;
    }
    if (!read || w == 0) {
        return;
    }
    for (uint32_t k = 0; k < kr_rows; ++k) {
        Runs::read(
            acc,
            (kstart + k) * hid_t,
            hstart + chunk_col0,
            hstart + chunk_col0 + w,
            l1_base + k * chunk_w * tile_bytes,
            tile_bytes);
    }
}

//: Hidden rows [k_lo, k_hi) of ONE phase-2 W_down K-block, into that block's slot. W_down's K axis
//: IS h's hidden axis, so the row index is the same linear index the gate/up N axis uses. The
//: reader takes the head rows and the writer the tail, so each side reads a contiguous run.
template <class Runs, class Acc>
FORCE_INLINE void read_wd_rows(
    const Acc& acc,
    uint32_t hbase,
    uint32_t k_lo,
    uint32_t k_hi,
    uint32_t out_col_start,
    uint32_t ec,
    uint32_t ec_max,
    uint32_t emb_t,
    uint32_t slot_base,
    uint32_t tile_bytes) {
    for (uint32_t k = k_lo; k < k_hi; ++k) {
        Runs::read(
            acc,
            (hbase + k) * emb_t,
            out_col_start,
            out_col_start + ec,
            slot_base + k * ec_max * tile_bytes,
            tile_bytes);
    }
}

// The reduce-scatter gather leg
//: Unicast MY slice of one accumulator into every column peer's landing CB, then signal each.
//:
//: `scatter_payload()` uses this core's CB cursor as the remote address proxy, valid ONLY when the
//: logical landing capacity equals the physical one. A phase-aliased CB has a larger LCM capacity,
//: so its caller must use this block-indexed form. The barrier before the signals is load-bearing.
FORCE_INLINE void scatter_payload_to(
    uint32_t rt_peers,
    uint32_t src_cb,
    uint32_t dst,
    uint32_t workers,
    uint32_t slice_tiles,
    uint32_t my_row,
    uint32_t tile_bytes) {
    const uint32_t src = get_read_ptr(src_cb);
    dst += my_row * slice_tiles * tile_bytes;
    const uint32_t bytes = slice_tiles * tile_bytes;
    for (uint32_t i = 0; i < workers; ++i) {
        const Peer p = peer_at(rt_peers, i);
        // Worker i takes the CONTIGUOUS tile range [i*a, (i+1)*a) of MY block — contiguous because
        // the gate/up layout is `m*HN_PAD + n`, so this is ONE transaction, not m_eff strided ones.
        // It lands at MY row's slot on every peer, which is where they all expect my contribution.
        noc_async_write(src + i * bytes, get_noc_addr(p.x, p.y, dst), bytes);
    }
    noc_async_write_barrier();
}

FORCE_INLINE void scatter_payload(
    uint32_t rt_peers,
    uint32_t src_cb,
    uint32_t dst_cb,
    uint32_t workers,
    uint32_t slice_tiles,
    uint32_t my_row,
    uint32_t tile_bytes) {
    scatter_payload_to(rt_peers, src_cb, get_write_ptr(dst_cb), workers, slice_tiles, my_row, tile_bytes);
}

//: Publish one completion per destination after its payload has landed.  This is separate from
//: `scatter_payload` so the reader and writer can retain concurrent up/gate transfers on the two
//: NoCs while one RISC emits a single readiness notification only after both legs are complete.
FORCE_INLINE void scatter_signal(uint32_t rt_peers, uint32_t sem_data, uint32_t workers) {
    const uint32_t sem = static_cast<uint32_t>(get_semaphore(sem_data));
    for (uint32_t i = 0; i < workers; ++i) {
        const Peer p = peer_at(rt_peers, i);
        noc_semaphore_inc(get_noc_addr(p.x, p.y, sem), 1);
    }
    noc_async_atomic_barrier();
}

FORCE_INLINE void scatter_leg(
    uint32_t rt_peers,
    uint32_t src_cb,
    uint32_t dst_cb,
    uint32_t sem_data,
    uint32_t workers,
    uint32_t slice_tiles,
    uint32_t my_row,
    uint32_t tile_bytes) {
    scatter_payload(rt_peers, src_cb, dst_cb, workers, slice_tiles, my_row, tile_bytes);
    scatter_signal(rt_peers, sem_data, workers);
}

}  // namespace moe_fused_swiglu
