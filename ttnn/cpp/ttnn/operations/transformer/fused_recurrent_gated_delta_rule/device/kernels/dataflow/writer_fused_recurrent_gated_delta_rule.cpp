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
// The state tiles are written as they arrive: the compute packs state_out in (i, j) order, so the writer waits for
// them cumulatively in groups of Vt (one i row) and issues each group's NoC writes at once, instead of waiting for all
// Kt*Vt tiles and then writing 64 KB after the last token. The CB memory is read by the NoC asynchronously, so a group
// is popped only after the token's single write barrier; state_out holds kv x 2 tiles, so the compute can pack the
// next token meanwhile.
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
    constexpr uint32_t group = Vt;  // state tiles per wait: one i row of the compute's (i, j) pack order

    Noc noc;
    CircularBuffer cbout(cb::out16);
    CircularBuffer cbst(cb::state_out);
    auto o_src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
    auto st_src = use<CircularBuffer::AddrSelector::READ_PTR>(cbst);

    // Token t's o tiles: wait for all Vt (the vector pass pushes them together), issue the writes.
    auto issue_o = [&](uint32_t t) {
        cbout.wait_front(Vt);
        for (uint32_t j = 0; j < Vt; j++) {
            noc.async_write(o_src, o_acc, o_tb, {.offset_bytes = j * o_tb}, {.page_id = (h * T + t) * Vt + j});
        }
    };
    // One state: kv tiles into pages base_page.., issued group by group as they become visible (cumulative waits; the
    // read pointer does not move until the pop after the barrier).
    auto issue_state = [&](uint32_t base_page) {
        for (uint32_t g0 = 0; g0 < kv; g0 += group) {
            const uint32_t n = (kv - g0 < group) ? (kv - g0) : group;
            cbst.wait_front(g0 + n);
            for (uint32_t i = 0; i < n; i++) {
                noc.async_write(
                    st_src, st_acc, page_tb, {.offset_bytes = (g0 + i) * page_tb}, {.page_id = base_page + g0 + i});
            }
        }
    };

    for (uint32_t t = 0; t < T; t++) {
        issue_o(t);  // o stays head-major
        if constexpr (per_token) {
            issue_state((t * BH + h) * kv);  // state is token-major
        }
        noc.async_write_barrier();
        cbout.pop_front(Vt);
        if constexpr (per_token) {
            cbst.pop_front(kv);
        }
    }
    if constexpr (!per_token) {
        issue_state(h * kv);  // final state only
        noc.async_write_barrier();
        cbst.pop_front(kv);
    }
}
