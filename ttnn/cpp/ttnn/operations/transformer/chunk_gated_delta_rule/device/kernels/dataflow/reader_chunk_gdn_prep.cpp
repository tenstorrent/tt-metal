// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (prep) reader: the constants (eye, tril, ones, masks) built in L1 once, then per-chunk q,k,v,g,beta.
// No initial state — the prep phase is state-independent. Device 2.0 API.
// The reads of one item are issued as one flight behind a single read barrier: the compute cannot start an
// item before q and k have landed, and at kickoff every producer of the fused program reads at once, so each
// extra barrier is a contended round trip. The constants are not read from DRAM: every producer would fetch
// the same few pages from one bank at kickoff.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#if defined(GDN_FUSED_PRODUCER)
#include "chunk_gdn_fused_map.hpp"
#endif

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

    // A work-item is a flat (head, chunk) index — exactly the DRAM tile-group index h*NC + c.
#if defined(GDN_FUSED_PRODUCER)
    // Fused producer p of the map (chunk_gdn_fused_map.hpp): its wi_count items in map order.
    const uint32_t p = get_arg_val<uint32_t>(0);
#else
    // Chunk-parallel: this core handles the work-items wi_start + n * wi_stride, n < wi_count.
    const uint32_t wi_start = get_arg_val<uint32_t>(0);
#endif
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    // Args 7..10: the DRAM copies of eye, tril, ones, masks; the constants are built in L1 instead.
    // Flat metadata (used by V_FLAT/QK_FLAT): NC = chunks/head, HV = value-head count, Hk = key-head count.
    const uint32_t NC = get_arg_val<uint32_t>(11);
    const uint32_t HV = get_arg_val<uint32_t>(12);
    const uint32_t Hk = get_arg_val<uint32_t>(13);
    // Cycles to wait before the first read: the fused factory's kickoff stagger, 0 for the phased prep.
    const uint32_t kickoff_wait_cycles = get_arg_val<uint32_t>(15);
#if defined(GDN_FUSED_PRODUCER)
    // The producer map: BH, then (NPH, NX, num, den) as in chunk_gdn_fused_map.hpp.
    const GdnFusedMap map{
        get_arg_val<uint32_t>(14),
        NC,
        get_arg_val<uint32_t>(16),
        get_arg_val<uint32_t>(17),
        get_arg_val<uint32_t>(18),
        get_arg_val<uint32_t>(19)};
    auto item_wi = [&](uint32_t n) {
        const GdnFusedItem it = gdn_fused_item(map, p, n);
        return it.h * NC + it.c;
    };
#else
    // Work-item stride: 1 for a contiguous slice (phased prep).
    const uint32_t wi_stride = get_arg_val<uint32_t>(14);
    auto item_wi = [&](uint32_t n) { return wi_start + n * wi_stride; };
#endif

    // Mixed precision: q/k/v are bf16; g/beta and the constants are fp32.
    const uint32_t tb_io = get_tile_size(cb_q);
    const uint32_t tb_f = get_tile_size(cb_g);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_f);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb_f);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;

    Noc noc;

    // Reserve `n` tiles of a CB and post their reads; no barrier, no push. The caller ends a flight with
    // `noc.async_read_barrier()` and pushes every CB it reserved.
    auto issue = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
    };
    auto publish = [](uint32_t cb_id, uint32_t n) { CircularBuffer(cb_id).push_back(n); };

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

    // One item's inputs, q and k first (the norm consumes them first), as one flight.
    auto issue_item = [&](uint32_t hc) {
        if constexpr (QK_FLAT) {
            issue_qk_flat(q_acc, cb_q, hc);
            issue_qk_flat(k_acc, cb_k, hc);
        } else {
            issue(q_acc, cb_q, hc * ck, ck, tb_io);
            issue(k_acc, cb_k, hc * ck, ck, tb_io);
        }
        if constexpr (V_FLAT) {
            issue_v_flat(hc);
        } else {
            issue(v_acc, cb_v, hc * cv, cv, tb_io);
        }
        issue(g_acc, cb_g, hc * Ct, Ct, tb_f);
        issue(b_acc, cb_beta, hc * Ct, Ct, tb_f);
    };
    auto publish_item = [&]() {
        publish(cb_q, ck);
        publish(cb_k, ck);
        publish(cb_v, cv);
        publish(cb_g, Ct);
        publish(cb_beta, Ct);
    };

    // Constants in L1: eye and tril (diagonal included) [Ct x Ct tiles], ones [Ct x Ct tiles], the quadrant masks
    // Qtl, Qbr, Q10 (tiles 0, 1, 2). An fp32 tile is four 16x16 faces: (0) rows 0-15 cols 0-15, (1) rows 0-15
    // cols 16-31, (2) rows 16-31 cols 0-15, (3) rows 16-31 cols 16-31. zero_constants: reserve and zero fill (own
    // barrier). fill_constants: one face of ones by RISC stores, NoC loopback copies of that face, the diagonal
    // patterns by RISC stores; the caller ends the copies with a read barrier and pushes the four CBs.
    constexpr uint32_t kFaceWords = 256, kFaceBytes = kFaceWords * 4, kOne = 0x3F800000u;
    CircularBuffer eye(cb_eye), tril(cb_tril), ones(cb_ones), mask(cb_mask);
    auto zero_constants = [&]() {
        eye.reserve_back(cc);
        tril.reserve_back(cc);
        ones.reserve_back(cc);
        mask.reserve_back(3);
        noc.async_write_zeros(eye, cc * tb_f);
        noc.async_write_zeros(tril, cc * tb_f);
        noc.async_write_zeros(mask, 3 * tb_f);
        noc.write_zeros_l1_barrier();
    };
    auto fill_constants = [&]() {
        const uint32_t ones_face = ones.get_write_ptr();
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_face);
        for (uint32_t i = 0; i < kFaceWords; i++) {
            w[i] = kOne;
        }
        for (uint32_t t = 0; t < Ct; t++) {
            volatile tt_l1_ptr uint32_t* e =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eye.get_write_ptr() + (t * Ct + t) * tb_f);
            volatile tt_l1_ptr uint32_t* l =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tril.get_write_ptr() + (t * Ct + t) * tb_f);
            for (uint32_t f = 0; f < 4 * kFaceWords; f += 3 * kFaceWords) {  // faces 0 and 3
                for (uint32_t r = 0; r < 16; r++) {
                    e[f + r * 17] = kOne;
                    for (uint32_t q = 0; q <= r; q++) {
                        l[f + r * 16 + q] = kOne;
                    }
                }
            }
        }
        asm volatile("fence");
        UnicastEndpoint self;
        const auto src = noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = ones_face};
        auto copy_face = [&](const CircularBuffer& cb, uint32_t tile, uint32_t face) {
            noc.async_read(self, cb, kFaceBytes, src, {.offset_bytes = tile * tb_f + face * kFaceBytes});
        };
        for (uint32_t t = 0; t < cc; t++) {
            for (uint32_t f = (t == 0) ? 1u : 0u; f < 4; f++) {
                copy_face(ones, t, f);
            }
        }
        for (uint32_t rt = 0; rt < Ct; rt++) {
            for (uint32_t ct = 0; ct < rt; ct++) {
                for (uint32_t f = 0; f < 4; f++) {
                    copy_face(tril, rt * Ct + ct, f);
                }
            }
            copy_face(tril, rt * Ct + rt, 2);
        }
        copy_face(mask, 0, 0);
        copy_face(mask, 1, 3);
        copy_face(mask, 2, 2);
    };
    auto publish_constants = [&]() {
        eye.push_back(cc);
        tril.push_back(cc);
        ones.push_back(cc);
        mask.push_back(3);
    };

    // Kickoff: the first item's reads in flight while the constants are filled; one barrier for both.
    zero_constants();
    if (kickoff_wait_cycles != 0) {
        riscv_wait(kickoff_wait_cycles);
    }
    {
        DeviceZoneScopedN("rd_kick");
        if (wi_count > 0) {
            issue_item(item_wi(0));
        }
        fill_constants();
        noc.async_read_barrier();
    }
    if (wi_count > 0) {
        publish_item();
    }
    publish_constants();

    for (uint32_t i = 1; i < wi_count; i++) {
        {
            DeviceZoneScopedN("rd_item");
            issue_item(item_wi(i));
            noc.async_read_barrier();
        }
        publish_item();
    }
}
