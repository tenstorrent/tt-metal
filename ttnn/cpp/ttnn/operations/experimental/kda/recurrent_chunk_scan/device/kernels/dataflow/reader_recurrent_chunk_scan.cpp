// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// KDA scan reader: the initial state S [K,V] once, then vector-decay prep
// intermediates v_beta, kd, q_decay, intra, k_dec_t, dl[K,1], t_inv. FP32 by default; selected intermediates may be
// BF16.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_dl = 11, cb_S = 8, cb_Tinv = 13;
constexpr uint32_t cb_vbeta = 17, cb_kd = 18, cb_qdecay = 19, cb_intra = 20, cb_kdec_t = 24;
constexpr uint32_t cb_summary_S = cb_qdecay;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);                  // per-core V-block width (tiles)
    constexpr uint32_t initial_state_mode = get_compile_time_arg_val(3);  // 0=provided; summary seeds are private
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);             // full V (tiles) for row stride
    constexpr bool summary_pair = get_compile_time_arg_val(6) == 1;
    (void)initial_state_mode;

    constexpr auto vb_a = TensorAccessorArgs<7>();
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

    const uint32_t vb_tb = get_tile_size(cb_vbeta);
    const uint32_t kd_tb = get_tile_size(cb_kd);
    const uint32_t qd_tb = get_tile_size(cb_qdecay);
    const uint32_t it_tb = get_tile_size(cb_intra);
    const uint32_t kc_tb = get_tile_size(cb_kdec_t);
    const uint32_t dl_tb = get_tile_size(cb_dl);
    const uint32_t ti_tb = get_tile_size(cb_Tinv);
    const uint32_t s0_tb = get_tile_size(cb_S);
    const auto vb_acc = TensorAccessor(vb_a, vb_addr, vb_tb);
    const auto kd_acc = TensorAccessor(kd_a, kd_addr, kd_tb);
    const auto qd_acc = TensorAccessor(qd_a, qd_addr, qd_tb);
    const auto it_acc = TensorAccessor(it_a, it_addr, it_tb);
    const auto kc_acc = TensorAccessor(kc_a, kc_addr, kc_tb);
    const auto dl_acc = TensorAccessor(dl_a, dl_addr, dl_tb);
    const auto ti_acc = TensorAccessor(ti_a, ti_addr, ti_tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, s0_tb);

    // V-independent tile counts (full reads). cv/kv are per-row Vt and handled by read_vslice.
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t kc = Kt * Ct;

    Noc noc;

    // Full (V-independent) read: n contiguous tiles from `base` into the CB.
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t tile_bytes, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tile_bytes, {.page_id = base + t}, {.offset_bytes = t * tile_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // V-slice read: R row-groups of Vt tiles each, laid out in DRAM with row stride Vt_full and
    // this core's column offset vb*Vt. Packs contiguously ([R, Vt]) into the CB. `row_base` is the
    // first-tile index of the tensor's [R, Vt_full] block for this (head[, chunk]).
    auto read_vslice = [&](const auto& acc, uint32_t cb_id, uint32_t tile_bytes, uint32_t row_base, uint32_t R) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(R * Vt);
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t src = row_base + r * Vt_full + vb * Vt;
            const uint32_t dstt = r * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_read(acc, cb, tile_bytes, {.page_id = src + vt}, {.offset_bytes = (dstt + vt) * tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb.push_back(R * Vt);
    };

    auto seed_identity = [&](CircularBuffer& cb) {
        constexpr uint32_t one_fp32 = 0x3F800000;
        constexpr uint32_t face_elements = 16 * 16;
        constexpr uint32_t tile_elements = 4 * face_elements;
        cb.reserve_back(Kt * Vt);
        volatile tt_l1_ptr uint32_t* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for (uint32_t index = 0; index < Kt * Vt * tile_elements; ++index) {
            state[index] = 0;
        }
        for (uint32_t local_col = 0; local_col < Vt; local_col++) {
            const uint32_t global_col = vb * Vt + local_col;
            if (global_col < Kt) {
                volatile tt_l1_ptr uint32_t* tile = state + (global_col * Vt + local_col) * tile_elements;
                for (uint32_t row = 0; row < 16; ++row) {
                    tile[row * 16 + row] = one_fp32;
                    tile[3 * face_elements + row * 16 + row] = one_fp32;
                }
            }
        }
        cb.push_back(Kt * Vt);
    };

    // The summary specialization seeds B from zero and A+B from identity on the same core.
    if constexpr (summary_pair) {
        CircularBuffer zero_cb(cb_S);
        zero_cb.reserve_back(Kt * Vt);
        noc.async_write_zeros(zero_cb, Kt * Vt * s0_tb);
        noc.write_zeros_l1_barrier();
        zero_cb.push_back(Kt * Vt);

        CircularBuffer identity_cb(cb_summary_S);
        seed_identity(identity_cb);
    } else {
        read_vslice(s0_acc, cb_S, s0_tb, h * Kt * Vt_full, Kt);
    }

    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t hc = h * NC + c;
        read_vslice(vb_acc, cb_vbeta, vb_tb, hc * Ct * Vt_full, Ct);  // v_beta [C, V] slice
        read_into(kd_acc, cb_kd, kd_tb, hc * ck, ck);                 // V-independent: full read
        if constexpr (!summary_pair) {
            read_into(qd_acc, cb_qdecay, qd_tb, hc * ck, ck);
            read_into(it_acc, cb_intra, it_tb, hc * cc, cc);
        }
        read_into(kc_acc, cb_kdec_t, kc_tb, hc * kc, kc);
        read_into(dl_acc, cb_dl, dl_tb, hc * Kt, Kt);
        read_into(ti_acc, cb_Tinv, ti_tb, hc * cc, cc);
    }
}
