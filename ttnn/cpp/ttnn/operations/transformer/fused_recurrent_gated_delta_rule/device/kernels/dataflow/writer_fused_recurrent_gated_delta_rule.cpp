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

    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t st_addr = get_arg_val<uint32_t>(3);
    const uint32_t BH = get_arg_val<uint32_t>(4);  // set by the program factory; token-major stride

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

    for (uint32_t t = 0; t < T; t++) {
        write_from(cbout, o_acc, (h * T + t) * cv, cv);  // o stays head-major
        if (per_token) {
            write_from(cbst, st_acc, (t * BH + h) * kv, kv);  // state is token-major
        }
    }
    if (!per_token) {
        write_from(cbst, st_acc, h * kv, kv);  // final state only
    }
}
