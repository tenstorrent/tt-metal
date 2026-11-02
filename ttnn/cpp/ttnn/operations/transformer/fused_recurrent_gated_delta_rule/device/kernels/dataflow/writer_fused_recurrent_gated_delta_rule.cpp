// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: o [1,V] per token; state [K,V] either per-token (verify slots) or once (final).
// o layout [BH*T,1,V] (head-major); final state [BH,K,V]. Device 2.0 API.
//
// Per-token state is written TOKEN-MAJOR: slab (t*BH + h), i.e. the [BH*T,K,V] buffer reads as
// [T,B,HV,K,V]. Head-major would scatter token t's heads at stride T, so the host had to untilize
// the whole state tensor, permute it and re-tilize -- per layer, per call. Token-major already IS
// the [B,T,HV,K,V] element order at B==1, so the host keeps one tensor and slices the accepted
// slot once at commit. This is the equivalent of the reference kernel writing its states straight
// into their ssm_state_indices slots.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 6, cb_state = 7;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(3);  // Vt is this core's share of V (QWEN36_1C_GDN_VSPLIT)

    constexpr auto o_a = TensorAccessorArgs<4>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t st_addr = get_arg_val<uint32_t>(3);
    const uint32_t BH = get_arg_val<uint32_t>(4);  // set by the program factory; token-major stride
    const uint32_t v0 = get_arg_val<uint32_t>(5);  // first V tile of this core's part

    const uint32_t tb = get_tile_size(cb_out);  // fp32; o and state share it
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    const auto st_acc = TensorAccessor(st_a, st_addr, tb);

    constexpr uint32_t cv = Vt;
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    CircularBuffer cbout(cb_out);
    CircularBuffer cbst(cb_state);

    auto write_from = [&](CircularBuffer& cb, const auto& acc, uint32_t base, uint32_t n) {
        cb.wait_front(n);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_write(src, acc, tb, {.offset_bytes = t * tb}, {.page_id = base + t});
        }
        noc.async_write_barrier();
        cb.pop_front(n);
    };

    // State block b: Kt rows of Vt_full tiles; this core owns tiles [v0, v0 + Vt) of each row (all of them unless split).
    auto write_state = [&](uint32_t block) {
        if constexpr (Vt == Vt_full) {
            write_from(cbst, st_acc, block * kv, kv);
        } else {
            cbst.wait_front(kv);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbst);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                for (uint32_t vt = 0; vt < Vt; vt++) {
                    noc.async_write(src, st_acc, tb, {.offset_bytes = (kt * Vt + vt) * tb},
                                    {.page_id = block * Kt * Vt_full + kt * Vt_full + v0 + vt});
                }
            }
            noc.async_write_barrier();
            cbst.pop_front(kv);
        }
    };
    for (uint32_t t = 0; t < T; t++) {
        write_from(cbout, o_acc, (h * T + t) * Vt_full + v0, cv);  // o stays head-major
        if constexpr (per_token) {
            write_state(t * BH + h);  // state is token-major
        }
    }
    if constexpr (!per_token) {
        write_state(h);  // final state only
    }
}
