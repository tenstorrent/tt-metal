// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (prep) writer: per chunk, drain the 7 state-independent intermediates to DRAM.
//   v_beta [C,V], kd [C,K], q_decay [C,K], intra [C,C], k_dec_t [K,C], dl [1 tile], t_inv [C,C].
// All fp32. Each DRAM tensor is [BH, NC, R, Col] TILE, so head h chunk c starts at
// tile (h*NC + c) * (tiles-per-chunk).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// CB indices (must match the prep compute kernel + program factory).
constexpr uint32_t cb_Tinv = 13, cb_vbeta = 14, cb_kd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_kdec_t = 24, cb_dl = 22;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    // Accessors in output order: v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv.
    constexpr auto vb_a = TensorAccessorArgs<3>();
    constexpr auto kd_a = TensorAccessorArgs<vb_a.next_compile_time_args_offset()>();
    constexpr auto qd_a = TensorAccessorArgs<kd_a.next_compile_time_args_offset()>();
    constexpr auto it_a = TensorAccessorArgs<qd_a.next_compile_time_args_offset()>();
    constexpr auto kc_a = TensorAccessorArgs<it_a.next_compile_time_args_offset()>();
    constexpr auto dl_a = TensorAccessorArgs<kc_a.next_compile_time_args_offset()>();
    constexpr auto ti_a = TensorAccessorArgs<dl_a.next_compile_time_args_offset()>();

    // Chunk-parallel: drain this core's contiguous work-item slice [wi_start, wi_start+wi_count).
    // Work-item index == flat DRAM tile-group index (h*NC + c), so no h/c needed here.
    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
    const uint32_t vb_addr = get_arg_val<uint32_t>(2);
    const uint32_t kd_addr = get_arg_val<uint32_t>(3);
    const uint32_t qd_addr = get_arg_val<uint32_t>(4);
    const uint32_t it_addr = get_arg_val<uint32_t>(5);
    const uint32_t kc_addr = get_arg_val<uint32_t>(6);
    const uint32_t dl_addr = get_arg_val<uint32_t>(7);
    const uint32_t ti_addr = get_arg_val<uint32_t>(8);

    const uint32_t tb = get_tile_size(cb_vbeta);  // all outputs are fp32 -> same tile size
    const auto vb_acc = TensorAccessor(vb_a, vb_addr, tb);
    const auto kd_acc = TensorAccessor(kd_a, kd_addr, tb);
    const auto qd_acc = TensorAccessor(qd_a, qd_addr, tb);
    const auto it_acc = TensorAccessor(it_a, it_addr, tb);
    const auto kc_acc = TensorAccessor(kc_a, kc_addr, tb);
    const auto dl_acc = TensorAccessor(dl_a, dl_addr, tb);
    const auto ti_acc = TensorAccessor(ti_a, ti_addr, tb);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kc = Kt * Ct;

    Noc noc;

    auto drain = [&](uint32_t cb_id, const auto& acc, uint32_t n, uint32_t chunk_base) {
        CircularBuffer cb(cb_id);
        cb.wait_front(n);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_write(src, acc, tb, {.offset_bytes = t * tb}, {.page_id = chunk_base + t});
        }
        noc.async_write_barrier();
        cb.pop_front(n);
    };

    // Drain roughly in the compute's push order (v_beta, t_inv, kd, intra, q_decay, k_dec_t, dl).
    for (uint32_t i = 0; i < wi_count; i++) {
        const uint32_t hc = wi_start + i;  // flat (head, chunk) index
        drain(cb_vbeta, vb_acc, cv, hc * cv);
        drain(cb_Tinv, ti_acc, cc, hc * cc);
        drain(cb_kd, kd_acc, ck, hc * ck);
        drain(cb_intra, it_acc, cc, hc * cc);
        drain(cb_qdecay, qd_acc, ck, hc * ck);
        drain(cb_kdec_t, kc_acc, kc, hc * kc);
        drain(cb_dl, dl_acc, 1, hc * 1);
    }
}
