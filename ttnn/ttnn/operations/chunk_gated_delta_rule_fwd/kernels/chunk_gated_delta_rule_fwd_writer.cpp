// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_fwd — writer (RISCV_0).
//
// Owns every L1 -> DRAM store of the program plus the semaphore INCREMENTS of the two segmented
// handoffs.  Per core, strictly in this order:
//
//   Stage P : per item — decay -> g_cumsum (scalar scatter), Tinv -> A (face-row scatter),
//             nkcd, Q, intra, P^T, Gamma_full, v_corr[:, vb] -> fp32 scratch;
//             write barrier, then +1 on sem_ready[seg(i)] of each of the NV scan cores of bh.
//   Stage S : per scan unit, per chunk — h_i -> h (full pages; skipped for i = 0 when the reader
//             copied initial_state through), v_new[:, vb] -> scratch; after the last chunk of a
//             segment: write barrier, +1 on sem_done[j] of every E core of (bh, segment j).
//             Then final_state.
//   Stage E : per item, per V block — o and v_new face-row scatter (rows t < T only).
//
// Drain order mirrors the compute kernel's push order exactly (the writer never inspects data).
// Every push/pop is the CB's single uniform quantum (QO / QF); only the valid tiles are written.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include "cgdr_common.hpp"

#pragma GCC optimize("Os")

namespace {

// Write a rows x cols tile block (packed row-major at src) into a row-major tile grid.
template <typename ACC>
FORCE_INLINE void write_tiles(
    const ACC& acc,
    uint32_t src,
    uint32_t base_tile,
    uint32_t rows,
    uint32_t row_stride,
    uint32_t cols,
    uint32_t tile_bytes) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(
                src + (r * cols + c) * tile_bytes, acc.get_noc_addr(base_tile + r * row_stride + c), tile_bytes);
        }
    }
}

// Pop one egress block after its writes have left L1.
FORCE_INLINE void release(uint32_t cb, uint32_t quantum) {
    noc_async_writes_flushed();
    cb_pop_front(cb, quantum);
}

template <typename ACC>
FORCE_INLINE void drain_scratch(const ACC& acc, uint32_t base_tile, uint32_t rows, uint32_t row_stride, uint32_t cols) {
    cb_wait_front(cb_scratch_egress, QF);
    write_tiles(acc, get_read_ptr(cb_scratch_egress), base_tile, rows, row_stride, cols, F32_TILE);
    release(cb_scratch_egress, QF);
}

// Face-row scatter of a [Ct, dn] tile block into a [B,T,H,dtot*32] output: rows t < T only.
template <typename ACC>
void scatter_rows(
    const ACC& acc, uint32_t src_base, uint32_t b, uint32_t t0, uint32_t dtot, uint32_t d0, uint32_t dn, uint32_t h) {
    const uint32_t dst_off = row_run0_bytes(h);
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        for (uint32_t j = 0; j < dn; ++j) {
            const uint32_t tile_addr = src_base + (ct * dn + j) * IN_TILE;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t t = t0 + ct * 32 + r;
                if (t >= gT) {
                    break;
                }
                const uint32_t page = (b * gT + t) * dtot + (d0 + j);
                const uint32_t s = tile_addr + row_run0_bytes(r);
                noc_async_write(s, acc.get_noc_addr(page, dst_off), ROW_RUN_BYTES);
                noc_async_write(
                    s + ROW_RUN_GAP_BYTES, acc.get_noc_addr(page, dst_off + ROW_RUN_GAP_BYTES), ROW_RUN_BYTES);
            }
        }
    }
}

// Scalar scatter of column 0 of a [Ct] tile column into column h of a [B,T,H] output.  A NoC write
// needs (l1_src & 15) == (dst & 15); column h is not 16-byte aligned, so each value is staged at a
// matching modulo in the writer-local cb_scalar_stage first.
template <typename ACC>
void scatter_scalars(const ACC& acc, uint32_t src_base, uint32_t wstage, uint32_t b, uint32_t t0, uint32_t h) {
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        const uint32_t tile_addr = src_base + ct * IN_TILE;
        const uint32_t rows = (gT - tbase) < 32 ? (gT - tbase) : 32;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t doff = tile_elem_off(r, h) * ESZ;
            const uint32_t slot = wstage + r * 16u + (doff & 15u);
            if constexpr (ESZ == 4) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot)[0] =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr)[tile_elem_off(r, 0)];
            } else {
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot)[0] =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr)[tile_elem_off(r, 0)];
            }
            noc_async_write(slot, acc.get_noc_addr(b * Tt + (tbase >> 5), doff), ESZ);
        }
        // The staging slots are reused by the next token tile.
        noc_async_writes_flushed();
    }
}

FORCE_INLINE void sem_inc(uint32_t x, uint32_t y, uint32_t sem_id) {
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1);
}

}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t sc_addr = get_arg_val<uint32_t>(a++);
    const uint32_t o_addr = get_arg_val<uint32_t>(a++);
    const uint32_t fs_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t vn_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gcs_addr = get_arg_val<uint32_t>(a++);
    const uint32_t A_addr = get_arg_val<uint32_t>(a++);
    const uint32_t num_items = get_arg_val<uint32_t>(a++);
    const uint32_t item_first = get_arg_val<uint32_t>(a++);
    const uint32_t item_stride = get_arg_val<uint32_t>(a++);
    const uint32_t num_units = get_arg_val<uint32_t>(a++);
    const uint32_t units_idx = a;
    a += num_units;
    const uint32_t ready_targets_idx = a;  // per item: NV (x, y) pairs
    a += num_items * NV * 2;
    uint32_t done_cursor = a;  // per unit, per segment: n, then n (x, y) pairs

    constexpr auto sca = TensorAccessorArgs<CT_ACC_BASE>();
    constexpr auto oa = TensorAccessorArgs<sca.next_compile_time_args_offset()>();
    constexpr auto fsa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<fsa.next_compile_time_args_offset()>();
    constexpr auto vna = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto gcsa = TensorAccessorArgs<vna.next_compile_time_args_offset()>();
    constexpr auto Aa = TensorAccessorArgs<gcsa.next_compile_time_args_offset()>();

    const auto sc_acc = TensorAccessor(sca, sc_addr, F32_TILE);
    const auto o_acc = TensorAccessor(oa, o_addr, IN_TILE);
    const auto fs_acc = TensorAccessor(fsa, fs_addr, IN_TILE);
    const auto h_acc = TensorAccessor(ha, h_addr, IN_TILE);
    const auto vn_acc = TensorAccessor(vna, vn_addr, IN_TILE);
    const auto gcs_acc = TensorAccessor(gcsa, gcs_addr, IN_TILE);
    const auto A_acc = TensorAccessor(Aa, A_addr, IN_TILE);

    const uint32_t wstage = get_write_ptr(cb_scalar_stage);

    // ------------------------------------------------------------------
    // Stage P
    // ------------------------------------------------------------------
    for (uint32_t r = 0; r < num_items; ++r) {
        const uint32_t wi = item_first + r * item_stride;
        const uint32_t i = wi / BH;
        const uint32_t bh = wi % BH;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = i * CHUNK;

        cb_wait_front(cb_out_egress, QO);  // decay -> g_cumsum
        scatter_scalars(gcs_acc, get_read_ptr(cb_out_egress), wstage, b, t0, h);
        release(cb_out_egress, QO);

        cb_wait_front(cb_out_egress, QO);  // Tinv -> A
        scatter_rows(A_acc, get_read_ptr(cb_out_egress), b, t0, Ct, 0, Ct, h);
        release(cb_out_egress, QO);

        drain_scratch(sc_acc, SC_NKCD + wi * CtKt, 1, 0, CtKt);
        drain_scratch(sc_acc, SC_QD + wi * CtKt, 1, 0, CtKt);
        drain_scratch(sc_acc, SC_INTRA + wi * CtCt, 1, 0, CtCt);
        drain_scratch(sc_acc, SC_PT + wi * CtKt, 1, 0, CtKt);
        drain_scratch(sc_acc, SC_GAM + wi, 1, 0, 1);
        for (uint32_t vb = 0; vb < NVI; ++vb) {
            drain_scratch(sc_acc, SC_VCORR + wi * CtVt + vb * Vi, Ct, Vt, Vi);
        }

        // P -> S handoff: the scratch of this item has LANDED before any scan core is told.
        noc_async_write_barrier();
        const uint32_t tbase = ready_targets_idx + r * NV * 2;
        for (uint32_t n = 0; n < NV; ++n) {
            sem_inc(
                get_arg_val<uint32_t>(tbase + 2 * n),
                get_arg_val<uint32_t>(tbase + 2 * n + 1),
                SEM_READY_BASE + seg_of(i));
        }
    }

    // ------------------------------------------------------------------
    // Stage S
    // ------------------------------------------------------------------
    for (uint32_t uu = 0; uu < num_units; ++uu) {
        const uint32_t u = get_arg_val<uint32_t>(units_idx + uu);
        const uint32_t bh = u / NV;
        const uint32_t vb = u % NV;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;

        for (uint32_t j = 0; j < NS; ++j) {
            for (uint32_t i = seg_begin(j); i < seg_end(j); ++i) {
                const uint32_t wi = i * BH + bh;
                if (!(HAS_H0 && i == 0)) {
                    cb_wait_front(cb_out_egress, QO);  // h_i
                    write_tiles(
                        h_acc,
                        get_read_ptr(cb_out_egress),
                        ((b * NC + i) * gH + h) * KtVt + vb * Vs,
                        Kt,
                        Vt,
                        Vs,
                        IN_TILE);
                    release(cb_out_egress, QO);
                }
                drain_scratch(sc_acc, SC_VNEW + wi * CtVt + vb * Vs, Ct, Vt, Vs);  // v_new
            }
            // S -> E handoff for segment j of this unit.
            noc_async_write_barrier();
            const uint32_t n = get_arg_val<uint32_t>(done_cursor++);
            for (uint32_t e = 0; e < n; ++e) {
                sem_inc(get_arg_val<uint32_t>(done_cursor), get_arg_val<uint32_t>(done_cursor + 1), SEM_DONE_BASE + j);
                done_cursor += 2;
            }
        }

        cb_wait_front(cb_out_egress, QO);  // final_state
        write_tiles(fs_acc, get_read_ptr(cb_out_egress), (b * gH + h) * KtVt + vb * Vs, Kt, Vt, Vs, IN_TILE);
        release(cb_out_egress, QO);
    }

    // ------------------------------------------------------------------
    // Stage E
    // ------------------------------------------------------------------
    for (uint32_t r = 0; r < num_items; ++r) {
        const uint32_t wi = item_first + r * item_stride;
        const uint32_t i = wi / BH;
        const uint32_t bh = wi % BH;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = i * CHUNK;
        for (uint32_t vb = 0; vb < NVI; ++vb) {
            cb_wait_front(cb_out_egress, QO);  // o
            scatter_rows(o_acc, get_read_ptr(cb_out_egress), b, t0, Vt, vb * Vi, Vi, h);
            release(cb_out_egress, QO);
            cb_wait_front(cb_out_egress, QO);  // v_new
            scatter_rows(vn_acc, get_read_ptr(cb_out_egress), b, t0, Vt, vb * Vi, Vi, h);
            release(cb_out_egress, QO);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
