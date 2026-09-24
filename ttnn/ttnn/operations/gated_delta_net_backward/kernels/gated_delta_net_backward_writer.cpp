// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gated_delta_net_backward — writer (RISCV_0 / BRISC).
//
// Owns every L1 -> DRAM store of the program plus the two semaphore
// INCREMENTS that release the stage barriers:
//
//   Stage P : drains the derived prep blocks to the flat f32 scratch, then
//             sends one `sem_prep` increment per item to that item's scan
//             owner (Barrier 1 fan-in, NC increments per group).
//   Stage S : drains S_i / v_new / dS / dv_new / dh0, then sends one
//             `sem_scan` increment to every core holding stage-G work in the
//             group it just scanned (Barrier 2 fan-out).
//   Stage G : the face-row SCATTER — the mirror of the reader's gather.  Only
//             rows t < T are written; the padded rows of the output tensors are
//             never touched and are sliced off by `to_torch`.
//
// Barrier ordering note (must be preserved): ALL prep increments are issued
// before ANY wait anywhere in the program, so the rendezvous cannot deadlock.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include "gdn_common.hpp"

// -Os: the three kernels of this op share ONE kernel-config ring buffer and the
// compute binaries dominate it, so every byte here counts.  These loops are
// NoC-issue bound, not instruction bound.
#pragma GCC optimize("Os")

constexpr uint32_t f32_tile = 32 * 32 * 4;
constexpr uint32_t in_tile = 32 * 32 * ESZ;

template <typename ACC>
static __attribute__((noipa)) void write_sub(
    const ACC& acc,
    uint32_t src,
    uint32_t base_tile,
    uint32_t rows,
    uint32_t row_stride,
    uint32_t col0,
    uint32_t cols,
    uint32_t tb) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(src + (r * cols + c) * tb, acc.get_noc_addr(base_tile + r * row_stride + col0 + c), tb);
        }
    }
}

// Pop one MAXBLK-sized egress block and write it to a flat scratch range.
template <typename ACC>
static __attribute__((noipa)) void drain_flat(const ACC& acc, uint32_t base_tile, uint32_t ntiles) {
    cb_wait_front(cb_egr, MAXBLK);
    const uint32_t src = get_read_ptr(cb_egr);
    for (uint32_t i = 0; i < ntiles; ++i) {
        noc_async_write(src + i * f32_tile, acc.get_noc_addr(base_tile + i), f32_tile);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_egr, MAXBLK);
}

template <typename ACC>
static __attribute__((noipa)) void drain_strided(
    const ACC& acc, uint32_t base_tile, uint32_t rows, uint32_t row_stride, uint32_t col0, uint32_t cols) {
    cb_wait_front(cb_egr, MAXBLK);
    write_sub(acc, get_read_ptr(cb_egr), base_tile, rows, row_stride, col0, cols, f32_tile);
    noc_async_write_barrier();
    cb_pop_front(cb_egr, MAXBLK);
}

// Face-row scatter of a [C, Dn] gradient block into a [B,T,H,D] output.
template <typename ACC>
static __attribute__((noipa)) void scatter_rows(
    const ACC& acc, uint32_t src_base, uint32_t b, uint32_t t0, uint32_t dtot, uint32_t d0, uint32_t dn, uint32_t h) {
    const uint32_t dst_off = row_run0_bytes(h);
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        for (uint32_t j = 0; j < dn; ++j) {
            const uint32_t tile_addr = src_base + (ct * dn + j) * in_tile;
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

// Scalar scatter of a [C, 1] gradient column into a [B,T,H] output.  The NoC
// requires (l1_src & 15) == (dram_dst & 15) for a DRAM write, and the element
// offset of column h is not 16-byte aligned, so each value is first staged at a
// matching modulo inside the (idle-by-now) gather scratch.
template <typename ACC>
static __attribute__((noipa)) void scatter_scalars(
    const ACC& acc, uint32_t src_base, uint32_t wstage, uint32_t b, uint32_t t0, uint32_t h) {
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        const uint32_t tt = tbase >> 5;
        const uint32_t tile_addr = src_base + ct * in_tile;
        uint32_t issued = 0;
        for (uint32_t r = 0; r < 32; ++r) {
            if (tbase + r >= gT) {
                break;
            }
            const uint32_t doff = tile_elem_off(r, h) * ESZ;
            const uint32_t slot = wstage + r * 16u + (doff & 15u);
            if constexpr (ESZ == 4) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
                d[0] = s[tile_elem_off(r, 0)];
            } else {
                volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
                volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot);
                d[0] = s[tile_elem_off(r, 0)];
            }
            noc_async_write(slot, acc.get_noc_addr(b * Tt + tt, doff), ESZ);
            ++issued;
        }
        if (issued) {
            noc_async_write_barrier();
        }
    }
}

void kernel_main() {
    const uint32_t sc_addr = get_arg_val<uint32_t>(0);
    const uint32_t dq_addr = get_arg_val<uint32_t>(1);
    const uint32_t dk_addr = get_arg_val<uint32_t>(2);
    const uint32_t dv_addr = get_arg_val<uint32_t>(3);
    const uint32_t dg_addr = get_arg_val<uint32_t>(4);
    const uint32_t dbeta_addr = get_arg_val<uint32_t>(5);
    const uint32_t dh0_addr = get_arg_val<uint32_t>(6);
    const uint32_t item_start = get_arg_val<uint32_t>(7);
    const uint32_t num_items = get_arg_val<uint32_t>(8);
    const uint32_t num_groups = get_arg_val<uint32_t>(11);
    const uint32_t group_base = 12;
    const uint32_t owned_cnt_idx = group_base + 3 * num_groups;
    const uint32_t num_owned = get_arg_val<uint32_t>(owned_cnt_idx);
    const uint32_t owned_base = owned_cnt_idx + 1;
    const uint32_t release_base = owned_base + num_owned;

    constexpr auto sca = TensorAccessorArgs<CT_ACC_BASE>();
    constexpr auto dqa = TensorAccessorArgs<sca.next_compile_time_args_offset()>();
    constexpr auto dka = TensorAccessorArgs<dqa.next_compile_time_args_offset()>();
    constexpr auto dva = TensorAccessorArgs<dka.next_compile_time_args_offset()>();
    constexpr auto dga = TensorAccessorArgs<dva.next_compile_time_args_offset()>();
    constexpr auto dba = TensorAccessorArgs<dga.next_compile_time_args_offset()>();
    constexpr auto dh0a = TensorAccessorArgs<dba.next_compile_time_args_offset()>();

    const auto sc_acc = TensorAccessor(sca, sc_addr, f32_tile);
    const auto dq_acc = TensorAccessor(dqa, dq_addr, in_tile);
    const auto dk_acc = TensorAccessor(dka, dk_addr, in_tile);
    const auto dv_acc = TensorAccessor(dva, dv_addr, in_tile);
    const auto dg_acc = TensorAccessor(dga, dg_addr, in_tile);
    const auto db_acc = TensorAccessor(dba, dbeta_addr, in_tile);
    [[maybe_unused]] const auto dh0_acc = TensorAccessor(dh0a, dh0_addr, in_tile);

    const uint32_t sem_prep_addr = get_semaphore(SEM_PREP);
    const uint32_t sem_scan_addr = get_semaphore(SEM_SCAN);

    // ------------------------------------------------------------------
    // Stage P — drain the prep blocks, then signal each item's scan owner.
    // ------------------------------------------------------------------
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;

        drain_flat(sc_acc, SC_ATTN + wi * ST_ATTN, CtCt);
        drain_flat(sc_acc, SC_KCD + wi * ST_KCD, CtKt);
        drain_flat(sc_acc, SC_P + wi * ST_P, CtKt);
        drain_flat(sc_acc, SC_VEC + wi * ST_VEC, 4 * Ct);

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            drain_strided(sc_acc, SC_VCORR + wi * ST_VCORR, Ct, Vt, vb * Vb, Vb);
            drain_strided(sc_acc, SC_U + wi * ST_U, Ct, Vt, vb * Vb, Vb);
            drain_strided(sc_acc, SC_C + wi * ST_C, Kt, Vt, vb * Vb, Vb);
        }

        // Barrier-1 fan-in: +1 on the owner of this item's group.
        const uint32_t bh = wi / NC;
        for (uint32_t gidx = 0; gidx < num_groups; ++gidx) {
            if (get_arg_val<uint32_t>(group_base + 3 * gidx) == bh) {
                const uint32_t ox = get_arg_val<uint32_t>(group_base + 3 * gidx + 1);
                const uint32_t oy = get_arg_val<uint32_t>(group_base + 3 * gidx + 2);
                noc_semaphore_inc(get_noc_addr(ox, oy, sem_prep_addr), 1);
                break;
            }
        }
    }
    noc_async_atomic_barrier();

    // ------------------------------------------------------------------
    // Stage S — drain the scan, then release the group.
    // ------------------------------------------------------------------
    {
        uint32_t rcursor = release_base;
        for (uint32_t oi = 0; oi < num_owned; ++oi) {
            const uint32_t bh = get_arg_val<uint32_t>(owned_base + oi);
            const uint32_t b = bh / gH;
            const uint32_t h = bh % gH;

            for (uint32_t vb = 0; vb < NVB; ++vb) {
                for (uint32_t i = 0; i < NC; ++i) {
                    const uint32_t wi = bh * NC + i;
                    drain_strided(sc_acc, SC_S + wi * ST_S, Kt, Vt, vb * Vb, Vb);
                    drain_strided(sc_acc, SC_VNEW + wi * ST_VNEW, Ct, Vt, vb * Vb, Vb);
                }
                for (int32_t i = (int32_t)NC - 1; i >= 0; --i) {
                    const uint32_t wi = bh * NC + (uint32_t)i;
                    drain_strided(sc_acc, SC_DS + wi * ST_DS, Kt, Vt, vb * Vb, Vb);
                    drain_strided(sc_acc, SC_DVNEW + wi * ST_DVNEW, Ct, Vt, vb * Vb, Vb);
                }
                if constexpr (HAS_H0) {
                    cb_wait_front(cb_gegr, MAXBLK_G);
                    const uint32_t src = get_read_ptr(cb_gegr);
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        for (uint32_t j = 0; j < Vb; ++j) {
                            noc_async_write(
                                src + (kt * Vb + j) * in_tile,
                                dh0_acc.get_noc_addr(((b * gH + h) * Kt + kt) * Vt + vb * Vb + j),
                                in_tile);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_gegr, MAXBLK_G);
                }
            }

            // Barrier-2 fan-out: +1 to every core with stage-G work in this group.
            const uint32_t ntargets = get_arg_val<uint32_t>(rcursor);
            ++rcursor;
            for (uint32_t t = 0; t < ntargets; ++t) {
                const uint32_t tx = get_arg_val<uint32_t>(rcursor + 2 * t);
                const uint32_t ty = get_arg_val<uint32_t>(rcursor + 2 * t + 1);
                noc_semaphore_inc(get_noc_addr(tx, ty, sem_scan_addr), 1);
            }
            rcursor += 2 * ntargets;
        }
        noc_async_atomic_barrier();
    }

    // ------------------------------------------------------------------
    // Stage G — the face-row scatter.
    // ------------------------------------------------------------------
    const uint32_t wstage = (get_write_ptr(cb_gather) + 63u) & ~63u;
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;
        const uint32_t bh = wi / NC;
        const uint32_t ci = wi % NC;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = ci * CHUNK;

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            cb_wait_front(cb_gegr, MAXBLK_G);
            scatter_rows(dv_acc, get_read_ptr(cb_gegr), b, t0, Vt, vb * Vb, Vb, h);
            noc_async_write_barrier();
            cb_pop_front(cb_gegr, MAXBLK_G);
        }

        // Drain order must mirror the compute kernel's push order exactly:
        // dv (per V block), dq, dbeta, dk, dg.
        cb_wait_front(cb_gegr, MAXBLK_G);
        scatter_rows(dq_acc, get_read_ptr(cb_gegr), b, t0, Kt, 0, Kt, h);
        noc_async_write_barrier();
        cb_pop_front(cb_gegr, MAXBLK_G);

        cb_wait_front(cb_gegr, MAXBLK_G);
        scatter_scalars(db_acc, get_read_ptr(cb_gegr), wstage, b, t0, h);
        cb_pop_front(cb_gegr, MAXBLK_G);

        cb_wait_front(cb_gegr, MAXBLK_G);
        scatter_rows(dk_acc, get_read_ptr(cb_gegr), b, t0, Kt, 0, Kt, h);
        noc_async_write_barrier();
        cb_pop_front(cb_gegr, MAXBLK_G);

        cb_wait_front(cb_gegr, MAXBLK_G);
        scatter_scalars(dg_acc, get_read_ptr(cb_gegr), wstage, b, t0, h);
        cb_pop_front(cb_gegr, MAXBLK_G);
    }
}
