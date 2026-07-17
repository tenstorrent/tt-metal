// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (prep) reader: constants (eye, tril, ones) once, then per-chunk q,k,v,g,beta.
// No initial state — the prep phase is state-independent. Device 2.0 API.

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

    // Mixed precision: q/k/v are bf16; g/beta and the constants are fp32.
    const uint32_t tb_io = get_tile_size(cb_q);
    const uint32_t tb_f = get_tile_size(cb_g);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_f);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb_f);
    const auto eye_acc = TensorAccessor(eye_a, eye_addr, tb_f);
    const auto tril_acc = TensorAccessor(tril_a, tril_addr, tb_f);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tb_f);
    const auto mask_acc = TensorAccessor(mask_a, mask_addr, tb_f);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    Noc noc;

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
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
    auto read_v_flat = [&](uint32_t hc) {
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
        noc.async_read_barrier();
        cbv.push_back(cv);
    };

    // Flat-q/k token-major read: work-item is value-head hv; its key-head is hk = hv / G (GQA group
    // size G = HV/Hk). Fetch [Ct,Kt] for (hk, chunk c) from the flat [B,T,Hk*K] grid (row stride Hk*Kt,
    // col offset hk*Kt), packed row-major into `cb` — identical layout to the head-major read.
    auto read_qk_flat = [&](const auto& acc, uint32_t cb_id, uint32_t hc) {
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
        noc.async_read_barrier();
        cb.push_back(ck);
    };

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
        read_into(g_acc, cb_g, hc * Ct, Ct, tb_f);
        read_into(b_acc, cb_beta, hc * Ct, Ct, tb_f);
    }
}
