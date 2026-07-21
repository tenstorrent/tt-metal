// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) reader: the initial state S [K,V] once, then per chunk the seven prep
// intermediates v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv from DRAM. All fp32.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_dl = 11, cb_S = 8, cb_Tinv = 13;
constexpr uint32_t cb_vbeta = 17, cb_kd = 18, cb_qdecay = 19, cb_intra = 20, cb_kdec_t = 24;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);  // per-core V-block width (tiles)
    constexpr uint32_t has_s0 = get_compile_time_arg_val(3);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);  // full V (tiles) for row stride
    (void)has_s0;

    constexpr auto vb_a = TensorAccessorArgs<5>();
    constexpr auto kd_a = TensorAccessorArgs<vb_a.next_compile_time_args_offset()>();
    constexpr auto qd_a = TensorAccessorArgs<kd_a.next_compile_time_args_offset()>();
    constexpr auto it_a = TensorAccessorArgs<qd_a.next_compile_time_args_offset()>();
    constexpr auto kc_a = TensorAccessorArgs<it_a.next_compile_time_args_offset()>();
    constexpr auto dl_a = TensorAccessorArgs<kc_a.next_compile_time_args_offset()>();
    constexpr auto ti_a = TensorAccessorArgs<dl_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<ti_a.next_compile_time_args_offset()>();

    // This core handles head h, V-block vb (columns [vb*Vt, vb*Vt+Vt) of the full V dimension).
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t vb_addr = get_arg_val<uint32_t>(3);
    const uint32_t kd_addr = get_arg_val<uint32_t>(4);
    const uint32_t qd_addr = get_arg_val<uint32_t>(5);
    const uint32_t it_addr = get_arg_val<uint32_t>(6);
    const uint32_t kc_addr = get_arg_val<uint32_t>(7);
    const uint32_t dl_addr = get_arg_val<uint32_t>(8);
    const uint32_t ti_addr = get_arg_val<uint32_t>(9);
    const uint32_t s0_addr = get_arg_val<uint32_t>(10);

    const uint32_t tb = get_tile_size(cb_vbeta);  // all inputs fp32 -> same tile size
    const auto vb_acc = TensorAccessor(vb_a, vb_addr, tb);
    const auto kd_acc = TensorAccessor(kd_a, kd_addr, tb);
    const auto qd_acc = TensorAccessor(qd_a, qd_addr, tb);
    const auto it_acc = TensorAccessor(it_a, it_addr, tb);
    const auto kc_acc = TensorAccessor(kc_a, kc_addr, tb);
    const auto dl_acc = TensorAccessor(dl_a, dl_addr, tb);
    const auto ti_acc = TensorAccessor(ti_a, ti_addr, tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb);

    // V-independent tile counts (full reads). cv/kv are per-row Vt and handled by read_vslice.
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t kc = Kt * Ct;

    Noc noc;

    // Full (V-independent) read: n contiguous tiles from `base` into the CB.
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // V-slice read: R row-groups of Vt tiles each, laid out in DRAM with row stride Vt_full and
    // this core's column offset vb*Vt. Packs contiguously ([R, Vt]) into the CB. `row_base` is the
    // first-tile index of the tensor's [R, Vt_full] block for this (head[, chunk]).
    auto read_vslice = [&](const auto& acc, uint32_t cb_id, uint32_t row_base, uint32_t R) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(R * Vt);
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t src = row_base + r * Vt_full + vb * Vt;
            const uint32_t dstt = r * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_read(acc, cb, tb, {.page_id = src + vt}, {.offset_bytes = (dstt + vt) * tb});
            }
        }
        noc.async_read_barrier();
        cb.push_back(R * Vt);
    };

    // initial state S [K, V] (once) — host always provides it (zeros if none). V-sliced.
    read_vslice(s0_acc, cb_S, h * Kt * Vt_full, Kt);

    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t hc = h * NC + c;
        read_vslice(vb_acc, cb_vbeta, hc * Ct * Vt_full, Ct);  // v_beta [C, V] slice
        read_into(kd_acc, cb_kd, hc * ck, ck);                 // V-independent: full read
        read_into(qd_acc, cb_qdecay, hc * ck, ck);
        read_into(it_acc, cb_intra, hc * cc, cc);
        read_into(kc_acc, cb_kdec_t, hc * kc, kc);
        read_into(dl_acc, cb_dl, hc * Kt, Kt);  // KDA: dl is per-K [K,1] = Kt tiles (was scalar 1 tile)
        read_into(ti_acc, cb_Tinv, hc * cc, cc);
    }
}
