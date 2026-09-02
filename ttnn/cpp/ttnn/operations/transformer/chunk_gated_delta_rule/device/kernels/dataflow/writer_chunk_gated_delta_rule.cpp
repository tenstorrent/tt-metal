// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: o [C,V] per chunk, then final_state [K,V] once. Device 2.0 API.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_final = 27;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr uint32_t has_s0 = get_compile_time_arg_val(3);
    (void)has_s0;

    constexpr auto o_a = TensorAccessorArgs<4>();
    constexpr auto fs_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t NC = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t fs_addr = get_arg_val<uint32_t>(3);

    // Mixed precision: o is bf16 (cb_out), the final state is fp32 (cb_final). Each write MUST use
    // its own tile size, else the fp32 state written at the bf16 stride is garbage (and vice versa).
    const uint32_t tb_o = get_tile_size(cb_out);
    const uint32_t tb_fs = get_tile_size(cb_final);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb_o);
    const auto fs_acc = TensorAccessor(fs_a, fs_addr, tb_fs);

    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    CircularBuffer cbout(cb_out);

    for (uint32_t c = 0; c < NC; c++) {
        cbout.wait_front(cv);
        const uint32_t base = (h * NC + c) * cv;
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
        for (uint32_t t = 0; t < cv; t++) {
            noc.async_write(src, o_acc, tb_o, {.offset_bytes = t * tb_o}, {.page_id = base + t});
        }
        noc.async_write_barrier();
        cbout.pop_front(cv);
    }

    // final_state
    CircularBuffer cbfs(cb_final);
    cbfs.wait_front(kv);
    const uint32_t base = h * kv;
    auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbfs);
    for (uint32_t t = 0; t < kv; t++) {
        noc.async_write(src, fs_acc, tb_fs, {.offset_bytes = t * tb_fs}, {.page_id = base + t});
    }
    noc.async_write_barrier();
    cbfs.pop_front(kv);
}
