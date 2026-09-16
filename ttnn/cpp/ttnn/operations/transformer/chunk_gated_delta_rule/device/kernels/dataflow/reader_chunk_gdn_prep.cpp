// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (prep) reader: constants (eye, tril, ones) once, then per-chunk q,k,v,g,beta.
// No initial state — the prep phase is state-independent. Device 2.0 API.
//
// Two modes, selected by the host: the default reads one work-item synchronously (reserve, read,
// barrier, push, per tensor), and GDN_PREP_PREFETCH (host env TT_GDN_PREP_PREFETCH=1) runs one
// work-item ahead so the DRAM latency overlaps compute. Both push exactly the same tiles in the
// same order; the prefetch mode needs the 2-slot input CBs the host allocates alongside the define.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7;
// Three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10) packed into one [1,1,32,96] tensor.
// Loaded once into the cb_u slot (17), which the stable-form prep no longer uses.
constexpr uint32_t cb_mask = 17;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    constexpr auto q_a = TensorAccessorArgs<3>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto eye_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto tril_a = TensorAccessorArgs<eye_a.next_compile_time_args_offset()>();
    constexpr auto ones_a = TensorAccessorArgs<tril_a.next_compile_time_args_offset()>();
    constexpr auto mask_a = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();
    // OPT-A: trailing compile args (after all TensorAccessorArgs). 1 => read that tensor FLAT token-major.
    constexpr uint32_t V_FLAT = get_compile_time_arg_val(mask_a.next_compile_time_args_offset());
    constexpr uint32_t QK_FLAT = get_compile_time_arg_val(mask_a.next_compile_time_args_offset() + 1);

    // Chunk-parallel: this core handles the contiguous work-item slice [wi_start, wi_start+wi_count).
    // A work-item is a flat (head, chunk) index; it is exactly the DRAM tile-group index (h*NC + c).
    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    const uint32_t eye_addr = get_arg_val<uint32_t>(7);
    const uint32_t tril_addr = get_arg_val<uint32_t>(8);
    const uint32_t ones_addr = get_arg_val<uint32_t>(9);
    const uint32_t mask_addr = get_arg_val<uint32_t>(10);
    // Flat metadata (used by V_FLAT/QK_FLAT): NC = chunks/head, HV = value-head count, Hk = key-head count.
    const uint32_t NC = get_arg_val<uint32_t>(11);
    const uint32_t HV = get_arg_val<uint32_t>(12);
    const uint32_t Hk = get_arg_val<uint32_t>(13);

    // Mixed precision: q/k/v are bf16; the constants (eye/tril/ones/masks) are always fp32; g/beta
    // are fp32 by default but bf16 under QWEN36_GDN_GB_BF16, so their tile stride comes from their
    // OWN CB (the host sets cb_g/cb_beta's format from the input dtype) — never from tb_f.
    const uint32_t tb_io = get_tile_size(cb_q);
    const uint32_t tb_f = get_tile_size(cb_eye);
    const uint32_t tb_gb = get_tile_size(cb_g);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_gb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb_gb);
    const auto eye_acc = TensorAccessor(eye_a, eye_addr, tb_f);
    const auto tril_acc = TensorAccessor(tril_a, tril_addr, tb_f);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tb_f);
    const auto mask_acc = TensorAccessor(mask_a, mask_addr, tb_f);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    Noc noc;

    // Reserve a slot in cb_id and issue its n tile reads into it. No barrier, no push: the caller
    // decides when to await them (the prefetch path awaits one work-item later).
    auto issue_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
    };

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        issue_into(acc, cb_id, base, n, tb);
        noc.async_read_barrier();
        CircularBuffer(cb_id).push_back(n);
    };

    // constants (once)
    read_into(eye_acc, cb_eye, 0, cc, tb_f);
    read_into(tril_acc, cb_tril, 0, cc, tb_f);
    read_into(ones_acc, cb_ones, 0, cc, tb_f);
    read_into(mask_acc, cb_mask, 0, 3, tb_f);  // Qtl, Qbr, Q10 (tiles 0,1,2)

    // Flat-v token-major read: fetch head hv's chunk c out of the flat [B,T,HV*V] tile grid
    // (row stride HV*Vt tiles, column offset hv*Vt), packing the [Ct,Vt] block contiguously into
    // cb_v in the SAME row-major order the head-major read produces (CB idx rt*Vt+ct) — so the
    // compute sees byte-identical tiles regardless of source layout. Requires pad==0 (T=NC*Ct*32).
    auto issue_v_flat = [&](uint32_t hc) {
        const uint32_t bh = hc / NC;
        const uint32_t c = hc % NC;
        const uint32_t hv = bh % HV;
        const uint32_t b = bh / HV;
        const uint32_t row_stride = HV * Vt;                   // tiles per token-row in flat v
        const uint32_t batch_base = b * NC * Ct * row_stride;  // b * (T/32) * row_stride
        CircularBuffer cbv(cb_v);
        cbv.reserve_back(cv);
        for (uint32_t rt = 0; rt < Ct; rt++) {
            for (uint32_t ct = 0; ct < Vt; ct++) {
                const uint32_t page = batch_base + (c * Ct + rt) * row_stride + hv * Vt + ct;
                noc.async_read(v_acc, cbv, tb_io, {.page_id = page}, {.offset_bytes = (rt * Vt + ct) * tb_io});
            }
        }
    };

    auto read_v_flat = [&](uint32_t hc) {
        issue_v_flat(hc);
        noc.async_read_barrier();
        CircularBuffer(cb_v).push_back(cv);
    };

    // Flat-q/k token-major read: work-item is value-head hv; its key-head is hk = hv / G (GQA group
    // size G = HV/Hk). Fetch [Ct,Kt] for (hk, chunk c) from the flat [B,T,Hk*K] grid (row stride Hk*Kt,
    // col offset hk*Kt), packed row-major into `cb` — identical layout to the head-major read.
    auto issue_qk_flat = [&](const auto& acc, uint32_t cb_id, uint32_t hc) {
        const uint32_t G = HV / Hk;
        const uint32_t bh = hc / NC;
        const uint32_t c = hc % NC;
        const uint32_t hv = bh % HV;
        const uint32_t b = bh / HV;
        const uint32_t hk = hv / G;
        const uint32_t row_stride = Hk * Kt;
        const uint32_t batch_base = b * NC * Ct * row_stride;
        CircularBuffer cb(cb_id);
        cb.reserve_back(ck);
        for (uint32_t rt = 0; rt < Ct; rt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                const uint32_t page = batch_base + (c * Ct + rt) * row_stride + hk * Kt + kt;
                noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = (rt * Kt + kt) * tb_io});
            }
        }
    };

    auto read_qk_flat = [&](const auto& acc, uint32_t cb_id, uint32_t hc) {
        issue_qk_flat(acc, cb_id, hc);
        noc.async_read_barrier();
        CircularBuffer(cb_id).push_back(ck);
    };

#ifdef GDN_PREP_PREFETCH
    // Prefetching reader (host: TT_GDN_PREP_PREFETCH=1, which also gives cb_q/k/v/g/beta 2 slots).
    // The synchronous form below stalls on DRAM latency five times per work-item while the compute
    // idles. Here work-item i+1's reads are issued into freshly reserved slots right after i is
    // pushed and are only awaited at the top of the next iteration, so that latency overlaps i's
    // compute. reserve_back for i+1 blocks until the compute popped i-1 — the intended flow control
    // (each of these CBs is popped exactly once per work-item, so 2 slots keep the ping-pong aligned).
    // Tiles pushed, their order within a push, and the per-unit push order (q, k, v, g, beta) are
    // unchanged, so the compute sees byte-identical input; only the timing differs.
    auto issue_unit = [&](uint32_t hc) {
        if constexpr (QK_FLAT) {
            issue_qk_flat(q_acc, cb_q, hc);
            issue_qk_flat(k_acc, cb_k, hc);
        } else {
            issue_into(q_acc, cb_q, hc * ck, ck, tb_io);
            issue_into(k_acc, cb_k, hc * ck, ck, tb_io);
        }
        if constexpr (V_FLAT) {
            issue_v_flat(hc);
        } else {
            issue_into(v_acc, cb_v, hc * cv, cv, tb_io);
        }
        issue_into(g_acc, cb_g, hc * Ct, Ct, tb_gb);
        issue_into(b_acc, cb_beta, hc * Ct, Ct, tb_gb);
    };

    if (wi_count > 0) {
        issue_unit(wi_start);  // work-item 0 is in flight before the loop
    }
    for (uint32_t i = 0; i < wi_count; i++) {
        noc.async_read_barrier();  // work-item i's q,k,v,g,beta have landed
        CircularBuffer(cb_q).push_back(ck);
        CircularBuffer(cb_k).push_back(ck);
        CircularBuffer(cb_v).push_back(cv);
        CircularBuffer(cb_g).push_back(Ct);
        CircularBuffer(cb_beta).push_back(Ct);
        if (i + 1 < wi_count) {
            issue_unit(wi_start + i + 1);
        }
    }
#else
    for (uint32_t i = 0; i < wi_count; i++) {
        const uint32_t hc = wi_start + i;  // flat (head, chunk) index
        if constexpr (QK_FLAT) {
            read_qk_flat(q_acc, cb_q, hc);
            read_qk_flat(k_acc, cb_k, hc);
        } else {
            read_into(q_acc, cb_q, hc * ck, ck, tb_io);
            read_into(k_acc, cb_k, hc * ck, ck, tb_io);
        }
        if constexpr (V_FLAT) {
            read_v_flat(hc);
        } else {
            read_into(v_acc, cb_v, hc * cv, cv, tb_io);
        }
        read_into(g_acc, cb_g, hc * Ct, Ct, tb_gb);
        read_into(b_acc, cb_beta, hc * Ct, Ct, tb_gb);
    }
#endif
}
