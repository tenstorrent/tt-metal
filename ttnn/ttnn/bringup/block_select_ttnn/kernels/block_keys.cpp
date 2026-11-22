// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.block_keys (block_select.py): one data-movement kernel. Per row r of the core's share:
//   out[r, i] = ids[r, pos[r, i] / B] B + pos[r, i] % B     (pos >= K B, e.g. 0xFFFFFFFF, or that id >= NBW:
//                                                          0xFFFFFFFF)
// pos [rows, KP] uint32 ROW_MAJOR (positions in block_gather's [rows, K B] row), ids [rows, K] uint32 ROW_MAJOR (the
// blocks block_gather read), out [rows, KP] uint32 ROW_MAJOR. compile-time args: cb, B, K, KP, NBW, GX, pos / ids /
// out accessors; common args: pos, ids, out addresses, per, extra (as block_gather).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t KP = get_compile_time_arg_val(3);
    constexpr uint32_t NBW = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);
    constexpr auto p_args = TensorAccessorArgs<6>();
    constexpr auto i_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t SENT = 0xFFFFFFFFu;
    static_assert(B >= 1 && B <= 32 && (B & (B - 1)) == 0, "B: a power of two <= 32");
    const auto pos = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(i_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2));
    const uint32_t per = get_common_arg_val<uint32_t>(3);
    const uint32_t extra = get_common_arg_val<uint32_t>(4);
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t count = per + (c < extra ? 1u : 0u);
    const uint32_t first = c * per + (c < extra ? c : extra);
    if (count == 0) {
        return;
    }
    cb_reserve_back(cb, 1);
    const uint32_t base = (get_write_ptr(cb) + 63u) & ~63u;
    const uint32_t pos_l1 = base;
    const uint32_t ids_l1 = pos_l1 + ((4u * KP + 63u) & ~63u);
    volatile tt_l1_ptr uint32_t* pv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    volatile tt_l1_ptr uint32_t* iv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
    for (uint32_t r = first; r < first + count; ++r) {
        noc_async_read(pos.get_noc_addr(r), pos_l1, 4u * KP);
        noc_async_read(ids.get_noc_addr(r), ids_l1, 4u * K);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < KP; ++i) {
            const uint32_t p = pv[i];
            uint32_t v = SENT;
            if (p < K * B) {
                const uint32_t id = iv[p / B];
                if (id < NBW) {
                    v = id * B + (p % B);
                }
            }
            pv[i] = v;  // (in place: the row is written back from the same buffer)
        }
        noc_async_write(pos_l1, out.get_noc_addr(r), 4u * KP);
        noc_async_write_barrier();
    }
}
