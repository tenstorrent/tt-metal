// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: o [1,V] per token; state [K,V] either per-token (verify slots) or once (final).
// o layout [BH*T,1,V] (head-major); final state [BH,K,V]. Device 2.0 API.
//
// o arrives as Vt tiles of 16x32 (2048 bytes, row 0 = o; see ../fused_recurrent_gdn_cb.hpp). The output tensor's
// pages are 32x32 fp32 tiles (4096 bytes) whose first 2048 bytes are faces 0 and 1 = rows 0-15, so each o tile is
// written at the start of its page; rows 16-31 of the page are padding the host never reads.
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

#include "../fused_recurrent_gdn_cb.hpp"

namespace cb = gdn_decode_cb;

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

    // Both output tensors are 32x32 fp32 tiles (4 KB pages); the o CB holds 16x32 tiles (2 KB).
    constexpr uint32_t page_tb = cb::tile_bytes(cb::tile_32x32);
    const uint32_t o_tb = get_tile_size(cb::out16);  // 2048 from the CB's declared tile dims
    const auto o_acc = TensorAccessor(o_a, o_addr, page_tb);
    const auto st_acc = TensorAccessor(st_a, st_addr, page_tb);

    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    CircularBuffer cbout(cb::out16);
    CircularBuffer cbst(cb::state_out);

    // n tiles of `tile_bytes` each from the CB front, one per page starting at page `base` (page offset 0).
    auto write_from = [&](CircularBuffer& cbuf, const auto& acc, uint32_t tile_bytes, uint32_t base, uint32_t n) {
        cbuf.wait_front(n);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbuf);
        for (uint32_t i = 0; i < n; i++) {
            noc.async_write(src, acc, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = base + i});
        }
        noc.async_write_barrier();
        cbuf.pop_front(n);
    };

    for (uint32_t t = 0; t < T; t++) {
        write_from(cbout, o_acc, o_tb, (h * T + t) * Vt, Vt);  // o stays head-major
        if constexpr (per_token) {
            write_from(cbst, st_acc, page_tb, (t * BH + h) * kv, kv);  // state is token-major
        }
    }
    if constexpr (!per_token) {
        write_from(cbst, st_acc, page_tb, h * kv, kv);  // final state only
    }
}
