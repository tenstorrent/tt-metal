// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_fwd — reader (RISCV_1).
//
// Owns every DRAM -> L1 load of the program plus the semaphore WAITS of the two segmented
// handoffs.  Per core, strictly in this order (the order is what makes the rendezvous
// deadlock-free — P never waits, S waits only on P, E only on S):
//
//   build_constant_tiles    EYE, LT, SL, SU [C,C], ONES [1,C], E_ROW0 — once per core
//   Stage P (items)         face-row span gather of q, k, v[:, vb]; whole-page read + column
//                           extract of g, beta; padded-tail zero fill
//   Stage S (scan units)    initial_state[:, vb] (and h[:, 0] = initial_state, copied through);
//                           per chunk nkcd, P^T, v_corr[:, vb], Gamma_full from scratch, after
//                           waiting sem_ready[segment]
//   Stage E (items)         Q, intra, h_i[:, vb], v_new[:, vb], after waiting sem_done[segment]
//
// THE FACE-ROW GATHER.  q is [B,T,H,K] tiled over (H,K), so a page is [32 heads x 32 dims] for ONE
// token and a head is ONE ROW of every page.  A tile row lives in two faces (16 elements each,
// exactly 256 elements apart), so one NoC read of a 272-element span covers the row.  The NoC
// requires the L1 destination to agree with the DRAM source modulo 64, hence every staging line
// starts at `stage64 + (src_off & 63)`.  The two runs are re-packed into the destination tile's
// face rows with RISC-V word copies while the next window's reads are in flight (one transaction
// id per staging slot).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/tensor_accessor.h"

#include "cgdr_common.hpp"

#pragma GCC optimize("Os")

namespace {

constexpr uint32_t ONE_F32 = 0x3F800000u;

// Zero `nbytes` of a CB starting `offset` bytes past its write pointer, with the DM engine
// (never a CPU store loop).
FORCE_INLINE void zero_cb(uint32_t cb, uint32_t offset, uint32_t nbytes) {
    Noc noc;
    CircularBuffer c(cb);
    noc.async_write_zeros(c, nbytes, {.offset_bytes = offset});
    noc.write_zeros_l1_barrier();
}

// Local L1 -> L1 copy through the NoC (own core).
FORCE_INLINE void l1_copy(uint32_t src, uint32_t dst, uint32_t nbytes) {
    noc_async_read(get_noc_addr(src), dst, nbytes);
}

static __attribute__((noipa)) void store_word(uint32_t addr, uint32_t elem, uint32_t bits) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr)[elem] = bits;
}

// ---------------------------------------------------------------------------
// build_constant_tiles — once per core.  Zero the whole block over the NoC, then write only the
// lanes that carry a pattern.  The all-ones tiles (ONES, and the off-diagonal blocks of LT / SL /
// SU) are produced by NoC doubling copies from one 16-word face row — no whole-tile CPU fill.
// ---------------------------------------------------------------------------
void build_constant_tiles() {
    cb_reserve_back(cb_const, NCONST);
    const uint32_t base = get_write_ptr(cb_const);
    zero_cb(cb_const, 0, NCONST * F32_TILE);

    // One all-ones tile at CST_ONES by doubling (16 words -> 4096 B).
    const uint32_t ones = base + CST_ONES * F32_TILE;
    for (uint32_t c = 0; c < 16; ++c) {
        store_word(ones, c, ONE_F32);
    }
    for (uint32_t sz = 64; sz < F32_TILE; sz <<= 1) {
        l1_copy(ones, ones + sz, sz);
        noc_async_read_barrier();
    }
    for (uint32_t t = 1; t < Ct; ++t) {
        l1_copy(ones, ones + t * F32_TILE, F32_TILE);
    }

    for (uint32_t ti = 0; ti < Ct; ++ti) {
        for (uint32_t si = 0; si < Ct; ++si) {
            const uint32_t t = ti * Ct + si;
            const uint32_t lt = base + (CST_LT + t) * F32_TILE;
            const uint32_t sl = base + (CST_SL + t) * F32_TILE;
            const uint32_t su = base + (CST_SU + t) * F32_TILE;
            const uint32_t ey = base + (CST_EYE + t) * F32_TILE;
            if (ti > si) {  // strictly below the diagonal block: LT = SL = 1
                l1_copy(ones, lt, F32_TILE);
                l1_copy(ones, sl, F32_TILE);
            } else if (ti < si) {  // strictly above: SU = 1
                l1_copy(ones, su, F32_TILE);
            } else {  // diagonal block: triangles and the identity, lane by lane
                for (uint32_t r = 0; r < 32; ++r) {
                    for (uint32_t c = 0; c < 32; ++c) {
                        const uint32_t e = tile_elem_off(r, c);
                        if (c <= r) {
                            store_word(lt, e, ONE_F32);
                        }
                        if (c < r) {
                            store_word(sl, e, ONE_F32);
                        }
                        if (c > r) {
                            store_word(su, e, ONE_F32);
                        }
                    }
                    store_word(ey, tile_elem_off(r, r), ONE_F32);
                }
            }
        }
    }
    const uint32_t erow = base + CST_EROW0 * F32_TILE;
    for (uint32_t c = 0; c < 32; ++c) {
        store_word(erow, tile_elem_off(0, c), ONE_F32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_const, NCONST);
}

FORCE_INLINE void copy_row_run(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
#pragma GCC unroll 16
    for (uint32_t i = 0; i < ROW_RUN_WORDS; ++i) {
        d[i] = s[i];
    }
}

// ---------------------------------------------------------------------------
// gather_block — [Ct, dn] tile block of head h, tokens [t0, t0 + C), d-tiles [d0, d0 + dn) of a
// [B,T,H,dtot*32] tensor, into `cb` (one push of `block_pages`).  Rows t >= T are zero.
// ---------------------------------------------------------------------------
template <typename ACC>
void gather_block(
    const ACC& acc,
    uint32_t cb,
    uint32_t b,
    uint32_t t0,
    uint32_t dtot,
    uint32_t d0,
    uint32_t dn,
    uint32_t h,
    uint32_t block_pages) {
    cb_reserve_back(cb, block_pages);
    const uint32_t dst_base = get_write_ptr(cb);
    if (t0 + CHUNK > gT) {
        // Padded tail: rows t >= T must read as zero (a whole chunk overwrites every row).
        zero_cb(cb, 0, Ct * dn * IN_TILE);
    }

    const uint32_t src_off = row_run0_bytes(h);
    const uint32_t stage64 = (get_write_ptr(cb_gather_stage) + 63u) & ~63u;
    const uint32_t line_shift = src_off & 63u;
    // A staging window is GATHER_TOKENS rows of ONE destination tile.
    constexpr uint32_t WPT = 32u / GATHER_TOKENS;
    const uint32_t nwin = Ct * WPT * dn;

    uint32_t issued[GATHER_DEPTH] = {0};
    for (uint32_t w = 0; w <= nwin; ++w) {
        if (w < nwin) {
            const uint32_t j = w % dn;
            const uint32_t sub = (w / dn) % WPT;
            const uint32_t ct = w / (dn * WPT);
            const uint32_t slot = w % GATHER_DEPTH;
            const uint32_t sbase = stage64 + slot * GATHER_SLOT_BYTES;
            uint32_t n = 0;
            noc_async_read_set_trid(1 + slot);
            for (uint32_t r = 0; r < GATHER_TOKENS; ++r) {
                const uint32_t t = t0 + ct * 32 + sub * GATHER_TOKENS + r;
                if (t >= gT) {
                    break;
                }
                const uint32_t page = (b * gT + t) * dtot + (d0 + j);
                noc_async_read(
                    acc.get_noc_addr(page, src_off), sbase + r * ROW_SPAN_STRIDE + line_shift, ROW_SPAN_BYTES);
                ++n;
            }
            issued[slot] = n;
        }
        if (w > 0) {
            const uint32_t pw = w - 1;
            const uint32_t pj = pw % dn;
            const uint32_t psub = (pw / dn) % WPT;
            const uint32_t pct = pw / (dn * WPT);
            const uint32_t pslot = pw % GATHER_DEPTH;
            const uint32_t psbase = stage64 + pslot * GATHER_SLOT_BYTES;
            noc_async_read_barrier_with_trid(1 + pslot);
            const uint32_t tile_addr = dst_base + (pct * dn + pj) * IN_TILE;
            for (uint32_t r = 0; r < issued[pslot]; ++r) {
                const uint32_t src = psbase + r * ROW_SPAN_STRIDE + line_shift;
                const uint32_t dst = tile_addr + row_run0_bytes(psub * GATHER_TOKENS + r);
                copy_row_run(dst, src);
                copy_row_run(dst + ROW_RUN_GAP_BYTES, src + ROW_RUN_GAP_BYTES);
            }
        }
    }
    noc_async_read_set_trid(0);
    cb_push_back(cb, block_pages);
}

// Column gather of a rank-3 gate tensor: head h is COLUMN h of page (b, t/32).  One whole-page read
// per token tile, then one scalar per row into column 0 of the destination tile (the consumers
// read column 0 only; the tile was zeroed first so every other lane is finite).
template <typename ACC>
void gather_gate(const ACC& acc, uint32_t dst_base, uint32_t b, uint32_t t0, uint32_t h) {
    const uint32_t stage64 = (get_write_ptr(cb_gather_stage) + 63u) & ~63u;
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        noc_async_read(acc.get_noc_addr(b * Tt + (tbase >> 5), 0), stage64, IN_TILE);
        noc_async_read_barrier();
        const uint32_t tile_addr = dst_base + ct * IN_TILE;
        const uint32_t rows = (gT - tbase) < 32 ? (gT - tbase) : 32;
        for (uint32_t r = 0; r < rows; ++r) {
            if constexpr (ESZ == 4) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr)[tile_elem_off(r, 0)] =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage64)[tile_elem_off(r, h)];
            } else {
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr)[tile_elem_off(r, 0)] =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage64)[tile_elem_off(r, h)];
            }
        }
    }
}

// Read a rows x cols tile sub-block of a row-major tile grid (row stride `row_stride`) into dst.
template <typename ACC>
FORCE_INLINE void read_tiles(
    const ACC& acc,
    uint32_t dst,
    uint32_t base_tile,
    uint32_t rows,
    uint32_t row_stride,
    uint32_t cols,
    uint32_t tile_bytes) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_read(
                acc.get_noc_addr(base_tile + r * row_stride + c), dst + (r * cols + c) * tile_bytes, tile_bytes);
        }
    }
}

FORCE_INLINE void sem_wait(uint32_t sem_id, uint32_t expected) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), expected);
}

}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    const uint32_t g_addr = get_arg_val<uint32_t>(a++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sc_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t num_items = get_arg_val<uint32_t>(a++);
    const uint32_t item_first = get_arg_val<uint32_t>(a++);
    const uint32_t item_stride = get_arg_val<uint32_t>(a++);
    const uint32_t num_units = get_arg_val<uint32_t>(a++);
    const uint32_t units_idx = a;
    a += num_units;
    const uint32_t exp_ready_idx = a;
    a += NS;
    const uint32_t exp_done_idx = a;

    constexpr auto qa = TensorAccessorArgs<CT_ACC_BASE>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto h0a = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto sca = TensorAccessorArgs<h0a.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<sca.next_compile_time_args_offset()>();

    const auto q_acc = TensorAccessor(qa, q_addr, IN_TILE);
    const auto k_acc = TensorAccessor(ka, k_addr, IN_TILE);
    const auto v_acc = TensorAccessor(va, v_addr, IN_TILE);
    const auto g_acc = TensorAccessor(ga, g_addr, IN_TILE);
    const auto b_acc = TensorAccessor(ba, beta_addr, IN_TILE);
    [[maybe_unused]] const auto h0_acc = TensorAccessor(h0a, h0_addr, IN_TILE);
    const auto sc_acc = TensorAccessor(sca, sc_addr, F32_TILE);
    const auto h_acc = TensorAccessor(ha, h_addr, IN_TILE);

    build_constant_tiles();

    // ------------------------------------------------------------------
    // Stage P — one (bh, chunk) item per iteration; items wi = first + r * stride, wi = i*BH + bh.
    // ------------------------------------------------------------------
    for (uint32_t r = 0; r < num_items; ++r) {
        const uint32_t wi = item_first + r * item_stride;
        const uint32_t i = wi / BH;
        const uint32_t bh = wi % BH;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = i * CHUNK;

        gather_block(q_acc, cb_q_in, b, t0, Kt, 0, Kt, h, CtKt);
        gather_block(k_acc, cb_k_in, b, t0, Kt, 0, Kt, h, CtKt);

        cb_reserve_back(cb_gate_in, 2 * Ct);
        {
            const uint32_t gbase = get_write_ptr(cb_gate_in);
            zero_cb(cb_gate_in, 0, 2 * Ct * IN_TILE);
            gather_gate(g_acc, gbase, b, t0, h);
            gather_gate(b_acc, gbase + Ct * IN_TILE, b, t0, h);
        }
        cb_push_back(cb_gate_in, 2 * Ct);

        for (uint32_t vb = 0; vb < NVI; ++vb) {
            gather_block(v_acc, cb_vblock_in, b, t0, Vt, vb * Vi, Vi, h, QV);
        }
    }

    // ------------------------------------------------------------------
    // Stage S — scan units u = bh * NV + vb.  Waits only on P (sem_ready).
    // ------------------------------------------------------------------
    for (uint32_t uu = 0; uu < num_units; ++uu) {
        const uint32_t u = get_arg_val<uint32_t>(units_idx + uu);
        const uint32_t bh = u / NV;
        const uint32_t vb = u % NV;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;

        if constexpr (HAS_H0) {
            cb_reserve_back(cb_vblock_in, QV);
            const uint32_t dst = get_write_ptr(cb_vblock_in);
            read_tiles(h0_acc, dst, (b * gH + h) * KtVt + vb * Vs, Kt, Vt, Vs, IN_TILE);
            noc_async_read_barrier();
            // h[:, 0] IS the initial state: copied through at its own dtype, bit-exact, and before
            // the push so it has landed before this unit's first segment is released.
            const uint32_t hbase = (b * NC * gH + h) * KtVt + vb * Vs;  // chunk 0
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                for (uint32_t j = 0; j < Vs; ++j) {
                    noc_async_write(dst + (kt * Vs + j) * IN_TILE, h_acc.get_noc_addr(hbase + kt * Vt + j), IN_TILE);
                }
            }
            noc_async_write_barrier();
            cb_push_back(cb_vblock_in, QV);
        }

        for (uint32_t j = 0; j < NS; ++j) {
            sem_wait(SEM_READY_BASE + j, get_arg_val<uint32_t>(exp_ready_idx + j));
            for (uint32_t i = seg_begin(j); i < seg_end(j); ++i) {
                const uint32_t wi = i * BH + bh;
                cb_reserve_back(cb_kmat_in, CtKt);
                cb_reserve_back(cb_scan_pt, CtKt);
                cb_reserve_back(cb_scan_vcorr, CtVs);
                cb_reserve_back(cb_scan_gamma, 1);
                read_tiles(sc_acc, get_write_ptr(cb_kmat_in), SC_NKCD + wi * CtKt, 1, 0, CtKt, F32_TILE);
                read_tiles(sc_acc, get_write_ptr(cb_scan_pt), SC_PT + wi * CtKt, 1, 0, CtKt, F32_TILE);
                read_tiles(sc_acc, get_write_ptr(cb_scan_vcorr), SC_VCORR + wi * CtVt + vb * Vs, Ct, Vt, Vs, F32_TILE);
                read_tiles(sc_acc, get_write_ptr(cb_scan_gamma), SC_GAM + wi, 1, 0, 1, F32_TILE);
                noc_async_read_barrier();
                cb_push_back(cb_kmat_in, CtKt);
                cb_push_back(cb_scan_pt, CtKt);
                cb_push_back(cb_scan_vcorr, CtVs);
                cb_push_back(cb_scan_gamma, 1);
            }
        }
    }

    // ------------------------------------------------------------------
    // Stage E — the P items again, ascending (hence ascending segment).  Waits only on S.
    // ------------------------------------------------------------------
    uint32_t cur_seg = 0xFFFFFFFFu;
    for (uint32_t r = 0; r < num_items; ++r) {
        const uint32_t wi = item_first + r * item_stride;
        const uint32_t i = wi / BH;
        const uint32_t bh = wi % BH;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t seg = seg_of(i);
        if (seg != cur_seg) {
            sem_wait(SEM_DONE_BASE + seg, get_arg_val<uint32_t>(exp_done_idx + seg));
            cur_seg = seg;
        }

        cb_reserve_back(cb_kmat_in, CtKt);
        cb_reserve_back(cb_intra_in, CtCt);
        read_tiles(sc_acc, get_write_ptr(cb_kmat_in), SC_QD + wi * CtKt, 1, 0, CtKt, F32_TILE);
        read_tiles(sc_acc, get_write_ptr(cb_intra_in), SC_INTRA + wi * CtCt, 1, 0, CtCt, F32_TILE);
        noc_async_read_barrier();
        cb_push_back(cb_kmat_in, CtKt);
        cb_push_back(cb_intra_in, CtCt);

        const uint32_t hbase = ((b * NC + i) * gH + h) * KtVt;
        for (uint32_t vb = 0; vb < NVI; ++vb) {
            cb_reserve_back(cb_vblock_in, QV);
            cb_reserve_back(cb_vnew_in, CtVi);
            read_tiles(h_acc, get_write_ptr(cb_vblock_in), hbase + vb * Vi, Kt, Vt, Vi, IN_TILE);
            read_tiles(sc_acc, get_write_ptr(cb_vnew_in), SC_VNEW + wi * CtVt + vb * Vi, Ct, Vt, Vi, F32_TILE);
            noc_async_read_barrier();
            cb_push_back(cb_vblock_in, QV);
            cb_push_back(cb_vnew_in, CtVi);
        }
    }
}
