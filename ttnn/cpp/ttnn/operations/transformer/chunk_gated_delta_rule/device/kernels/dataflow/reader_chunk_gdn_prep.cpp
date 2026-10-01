// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (prep) reader: constants (eye, tril, ones) once, then per-chunk q,k,v,g,beta.
// No initial state — the prep phase is state-independent. Device 2.0 API.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include <utility>

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7;
// Three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10) packed into one [1,1,32,96] tensor.
// Loaded once into the cb_u slot (17), which the stable-form prep no longer uses.
constexpr uint32_t cb_mask = 17;
// gb_flat (Option B, fused path only): this core's one-hot head selector is loaded ONCE as tile 3
// of cb_mask (the fused u slot holds max(cv,3)+1 tiles with the credit tile last, so tile 3 is free
// when cv >= 4) — no extra CB, so the CB region does not grow. Valid
// because every fused producer core serves exactly one head (wi = h*NC + j + i*NP).

// Compile-time-unrolled one-word stores: constant offsets, so each store is a single instruction.
// i-th element (row-major) of the lower triangle (diagonal included) of a 16x16 face.
constexpr uint32_t tril_face_index(uint32_t i) {
    uint32_t r = 0;
    while ((r + 1) * (r + 2) / 2 <= i) {
        ++r;
    }
    return r * 16 + (i - r * (r + 1) / 2);
}
template <uint32_t... I>
inline __attribute__((always_inline)) void store_tril_lower(
    volatile tt_l1_ptr uint32_t* face, std::integer_sequence<uint32_t, I...>) {
    ((face[tril_face_index(I)] = 0x3F800000u), ...);
}
template <uint32_t... R>
inline __attribute__((always_inline)) void store_diag(
    volatile tt_l1_ptr uint32_t* face, std::integer_sequence<uint32_t, R...>) {
    ((face[R * 17] = 0x3F800000u), ...);
}

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
    // gb_flat: sel's accessor is ALWAYS present (a null-buffer stub when gb_flat is off — see the
    // program factory), so this offset never shifts between gb_flat on/off.
    constexpr auto sel_a = TensorAccessorArgs<mask_a.next_compile_time_args_offset()>();
    // OPT-A/gb_flat: trailing compile args (after all TensorAccessorArgs). 1 => read that tensor
    // FLAT token-major (V_FLAT/QK_FLAT) or select head h's column in-kernel (GB_FLAT).
    constexpr uint32_t V_FLAT = get_compile_time_arg_val(sel_a.next_compile_time_args_offset());
    constexpr uint32_t QK_FLAT = get_compile_time_arg_val(sel_a.next_compile_time_args_offset() + 1);
    constexpr uint32_t GB_FLAT = get_compile_time_arg_val(sel_a.next_compile_time_args_offset() + 2);

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
    // Work-item stride: 1 for a contiguous slice (phased prep), NP for the fused NP>1 producer
    // split, where producer p of head h owns the interleaved chunks c = p, p+NP, ... (wi stays the
    // flat h*NC + c, so every DRAM index below is unchanged).
    const uint32_t wi_stride = get_arg_val<uint32_t>(14);
    // gb_flat only (garbage/unused otherwise): the one-hot head-selector tensor's address. Appended
    // last so it never shifts the indices above.
    const uint32_t sel_addr = get_arg_val<uint32_t>(15);

    // Mixed precision: q/k/v are bf16; g/beta and the constants are fp32.
    const uint32_t tb_io = get_tile_size(cb_q);
#if defined(GDN_QK_FP32)
    const uint32_t tb_v = get_tile_size(cb_v);  // qk_prenormed fp32 q/k (fused factory): q/k fp32, v bf16
#else
    const uint32_t tb_v = tb_io;
#endif
    const uint32_t tb_f = get_tile_size(cb_g);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_v);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_f);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb_f);
    const auto eye_acc = TensorAccessor(eye_a, eye_addr, tb_f);
    const auto tril_acc = TensorAccessor(tril_a, tril_addr, tb_f);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tb_f);
    const auto mask_acc = TensorAccessor(mask_a, mask_addr, tb_f);
    const auto sel_acc = TensorAccessor(sel_a, sel_addr, tb_f);  // gb_flat only; unused otherwise

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

    // Flat-v token-major read: fetch head hv's chunk c out of the flat [B,T,HV*V] tile grid
    // (row stride HV*Vt tiles, column offset hv*Vt), packing the [Ct,Vt] block contiguously into
    // cb_v in the SAME row-major order the head-major read produces (CB idx rt*Vt+ct) — so the
    // compute sees byte-identical tiles regardless of source layout. Requires pad==0 (T=NC*Ct*32).
    auto read_v_flat = [&](uint32_t hc, bool wait) {
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
                noc.async_read(v_acc, cbv, tb_v, {.page_id = page}, {.offset_bytes = (rt * Vt + ct) * tb_v});
            }
        }
        if (wait) {
            noc.async_read_barrier();
            cbv.push_back(cv);
        }
    };

    // Flat-q/k token-major read: work-item is value-head hv; its key-head is hk = hv / G (GQA group
    // size G = HV/Hk). Fetch [Ct,Kt] for (hk, chunk c) from the flat [B,T,Hk*K] grid (row stride Hk*Kt,
    // col offset hk*Kt), packed row-major into `cb` — identical layout to the head-major read.
    auto read_qk_flat = [&](const auto& acc, uint32_t cb_id, uint32_t hc, bool wait) {
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
        if (wait) {
            noc.async_read_barrier();
            cb.push_back(ck);
        }
    };

    // gb_flat (Option B) raw read: fetch Ct tiles of the model's native [B,T,HV] g/beta tensor at
    // this work item's chunk (tile-col 0 — HV<=32 is asserted host-side, so the tensor is exactly
    // one tile wide). Every head sharing this (batch, chunk) reads the IDENTICAL Ct pages (no head
    // index in the addressing at all) — the per-head selection happens in compute via the
    // selector in cb_mask tile 3. Mirrors read_v_flat's page-range technique, minus the head offset.
    auto read_gb_flat = [&](const auto& acc, uint32_t cb_id, uint32_t hc, bool wait) {
        const uint32_t bh = hc / NC;
        const uint32_t c = hc % NC;
        const uint32_t b = bh / HV;
        const uint32_t batch_base = b * NC * Ct;  // Wt=1 tile-col => row stride is 1 tile
        CircularBuffer cb(cb_id);
        cb.reserve_back(Ct);
        for (uint32_t rt = 0; rt < Ct; rt++) {
            const uint32_t page = batch_base + c * Ct + rt;
            noc.async_read(acc, cb, tb_f, {.page_id = page}, {.offset_bytes = rt * tb_f});
        }
        if (wait) {
            noc.async_read_barrier();
            cb.push_back(Ct);
        }
    };

    // P15 C1 (GDN_COLD_PREFETCH): item 0's input reads overlap the constant build instead of following it. g/beta
    // (needed first: the head select) are issued before the constants and covered by the constants' own read barrier;
    // v/k/q (needed >= 2 us into the item) are issued after the constants are pushed, so the first item's compute
    // starts as soon as the constants and g/beta are in. Production flat layout only.
    constexpr bool kColdPrefetch =
#ifdef GDN_COLD_PREFETCH
        (Ct == 1) && QK_FLAT && V_FLAT && GB_FLAT;
#else
        false;
#endif
    const bool prefetched = kColdPrefetch && (wi_count > 0);

    // constants (once)
    if constexpr (Ct == 1) {
        // P3_FLAPREP: at chunk 32 every constant is a fixed 0/1 fp32 pattern (the host builds exactly these:
        // build_fused_const_tiles / make_head_selectors), so build them in L1 instead of reading them from DRAM:
        // every producer core read the same few DRAM pages here (~23 us per call at BH=16, NP=5). The zero words come
        // from the MEM_ZEROS loopback, one face of ones is replicated by local NoC copies, and the remaining one-words
        // are single stores (fully unrolled: constant offsets).
        constexpr uint32_t kOne = 0x3F800000u;  // 1.0f
        constexpr uint32_t kTileBytes = 4096;   // fp32 32x32 = 4 faces of 16x16
        constexpr uint32_t kFaceBytes = 1024;
        constexpr uint32_t n_mask = GB_FLAT ? 4 : 3;
#if defined(GDN_RD_NO_QMASKS)
        // GDN_TINV_SFPU producers never read the Horner quadrant masks (mask tiles 0-2); only the head selector (tile
        // 3).
        static_assert(GB_FLAT, "GDN_RD_NO_QMASKS needs the gb_flat head selector in mask tile 3");
        constexpr bool kQMasks = false;
#else
        constexpr bool kQMasks = true;
#endif
        CircularBuffer c_eye(cb_eye), c_tril(cb_tril), c_ones(cb_ones), c_mask(cb_mask);
        c_eye.reserve_back(1);
        c_tril.reserve_back(1);
        c_ones.reserve_back(1);
        c_mask.reserve_back(n_mask);
        const uint32_t eye_l1 = c_eye.get_write_ptr();
        const uint32_t tril_l1 = c_tril.get_write_ptr();
        const uint32_t ones_l1 = c_ones.get_write_ptr();
        const uint32_t mask_l1 = c_mask.get_write_ptr();
        if constexpr (kColdPrefetch) {
            if (prefetched) {
                read_gb_flat(g_acc, cb_g, wi_start, false);
                read_gb_flat(b_acc, cb_beta, wi_start, false);
            }
        }
        // Zero-fill (loopback reads). tril face 2 (all ones) is a NoC copy below, so it is not zeroed: the copy needs
        // no barrier against the zero-fill.
        noc.async_write_zeros(c_eye, kTileBytes);
        noc.async_write_zeros(c_tril, 2 * kFaceBytes);                                // faces 0, 1
        noc.async_write_zeros(c_tril, kFaceBytes, {.offset_bytes = 3 * kFaceBytes});  // face 3
        if constexpr (kQMasks) {
            noc.async_write_zeros(c_mask, n_mask * kTileBytes);
        } else {
            noc.async_write_zeros(c_mask, kTileBytes, {.offset_bytes = 3 * kTileBytes});  // selector tile only
        }
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_l1);
#pragma GCC unroll 256
        for (uint32_t i = 0; i < kFaceBytes / 4; i++) {
            ones[i] = kOne;  // face 0 of the ones tile, while the zero-fill runs
        }
        (void)ones[kFaceBytes / 4 - 1];  // read back: the stores have landed before the NoC reads the face
        // Replicate the ones face: ones faces 1-3, tril face 2 (and, unless skipped, Qtl face 0, Qbr face 3, Q10 face
        // 2).
        const uint64_t ones_face = get_noc_addr(ones_l1);
        noc_async_read(ones_face, ones_l1 + 1 * kFaceBytes, kFaceBytes);
        noc_async_read(ones_face, ones_l1 + 2 * kFaceBytes, kFaceBytes);
        noc_async_read(ones_face, ones_l1 + 3 * kFaceBytes, kFaceBytes);
        noc_async_read(ones_face, tril_l1 + 2 * kFaceBytes, kFaceBytes);
        if constexpr (kQMasks) {
            noc_async_read(ones_face, mask_l1 + 0 * kTileBytes + 0 * kFaceBytes, kFaceBytes);
            noc_async_read(ones_face, mask_l1 + 1 * kTileBytes + 3 * kFaceBytes, kFaceBytes);
            noc_async_read(ones_face, mask_l1 + 2 * kTileBytes + 2 * kFaceBytes, kFaceBytes);
        }
        noc.async_read_barrier();  // zero-fill, copies and (cold prefetch) g/beta have landed
        // The first item's first stage needs only the head selector and g/beta: push them now, the rest of the
        // constants follow (the compute waits for tril / ones at its second stage, for eye at its fifth).
        if constexpr (GB_FLAT) {
            // Head selector: one-hot at (row hv, col 0) — make_head_selectors' tile hv (face 0 or 2).
            const uint32_t hv = (wi_start / NC) % HV;
            volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1 + 3 * kTileBytes);
#if defined(GDN_SEL_BC)
            // GDN_SEL_BC: the whole ROW hv is ones (faces (hv>>4)*2 and (hv>>4)*2+1): g @ sel then puts head hv's g in
            // EVERY column of the selected tile instead of column 0 only; column 0 is bit-identical to the one-hot
            // selector.
            volatile tt_l1_ptr uint32_t* row0 = sel + ((hv >> 4) << 9) + ((hv & 15) << 4);
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                row0[c] = kOne;
                row0[256 + c] = kOne;
            }
#else
            sel[((hv >> 4) << 9) + ((hv & 15) << 4)] = kOne;
#endif
        }
        c_mask.push_back(n_mask);
        if constexpr (kColdPrefetch) {
            if (prefetched) {
                CircularBuffer(cb_g).push_back(Ct);
                CircularBuffer(cb_beta).push_back(Ct);
            }
        }
        // Stores into words no copy touches: the lower triangles of tril faces 0 and 3, the eye diagonal (faces 0 and
        // 3). The zero-fill has landed (barrier above).
        {
            volatile tt_l1_ptr uint32_t* tril = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tril_l1);
            volatile tt_l1_ptr uint32_t* eye = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eye_l1);
            store_tril_lower(tril, std::make_integer_sequence<uint32_t, 136>{});
            store_tril_lower(tril + 768, std::make_integer_sequence<uint32_t, 136>{});
            store_diag(eye, std::make_integer_sequence<uint32_t, 16>{});
            store_diag(eye + 768, std::make_integer_sequence<uint32_t, 16>{});
        }
        c_tril.push_back(1);
        c_ones.push_back(1);
        c_eye.push_back(1);
        if constexpr (kColdPrefetch) {
            if (prefetched) {
                // v, k, q of item 0 (needed in that order), landing while the first item's compute starts.
                read_v_flat(wi_start, false);
                read_qk_flat(k_acc, cb_k, wi_start, false);
                read_qk_flat(q_acc, cb_q, wi_start, false);
                noc.async_read_barrier();
                CircularBuffer(cb_v).push_back(cv);
                CircularBuffer(cb_k).push_back(ck);
                CircularBuffer(cb_q).push_back(ck);
            }
        }
    } else {
        read_into(eye_acc, cb_eye, 0, cc, tb_f);
        read_into(tril_acc, cb_tril, 0, cc, tb_f);
        read_into(ones_acc, cb_ones, 0, cc, tb_f);
        read_into(mask_acc, cb_mask, 0, 3, tb_f);  // Qtl, Qbr, Q10 (tiles 0,1,2)
        if constexpr (GB_FLAT) {
            // Selector tile hv (page hv of the single-tile-row [1,1,32,32*HV] tensor) -> cb_mask tile 3.
            const uint32_t hv = (wi_start / NC) % HV;
            read_into(sel_acc, cb_mask, hv, 1, tb_f);
        }
    }

    for (uint32_t i = prefetched ? 1 : 0; i < wi_count; i++) {
        const uint32_t hc = wi_start + i * wi_stride;  // flat (head, chunk) index
        if constexpr (QK_FLAT) {
            read_qk_flat(q_acc, cb_q, hc, true);
            read_qk_flat(k_acc, cb_k, hc, true);
        } else {
            read_into(q_acc, cb_q, hc * ck, ck, tb_io);
            read_into(k_acc, cb_k, hc * ck, ck, tb_io);
        }
        if constexpr (V_FLAT) {
            read_v_flat(hc, true);
        } else {
            read_into(v_acc, cb_v, hc * cv, cv, tb_v);
        }
        if constexpr (GB_FLAT) {
            read_gb_flat(g_acc, cb_g, hc, true);
            read_gb_flat(b_acc, cb_beta, hc, true);
        } else {
            read_into(g_acc, cb_g, hc * Ct, Ct, tb_f);
            read_into(b_acc, cb_beta, hc * Ct, Ct, tb_f);
        }
    }
}
