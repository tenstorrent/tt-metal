// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) writer, value-parallel. This core produced ONE V-block (vb) of head h:
// columns [vb*Vt, vb*Vt+Vt) of the full V dimension. It writes that slice back into the full
// tensors o [BH, NC, C, V] and final_state [BH, K, V] using DRAM row stride Vt_full.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_final = 27;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);  // per-core V-block width (tiles)
    constexpr uint32_t has_s0 = get_compile_time_arg_val(3);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);  // full V (tiles) for row stride
    (void)has_s0;

    constexpr auto o_a = TensorAccessorArgs<5>();
    constexpr auto fs_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t fs_addr = get_arg_val<uint32_t>(4);

    // Mixed precision: o is bf16 (cb_out), the final state is fp32 (cb_final). Each write MUST use
    // its own tile size, else the fp32 state written at the bf16 stride is garbage (and vice versa).
    const uint32_t tb_o = get_tile_size(cb_out);
    const uint32_t tb_fs = get_tile_size(cb_final);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb_o);
    const auto fs_acc = TensorAccessor(fs_a, fs_addr, tb_fs);

    constexpr uint32_t cv = Ct * Vt;  // per-core [C, Vt] output slab
    constexpr uint32_t kv = Kt * Vt;  // per-core [K, Vt] final-state slab

    Noc noc;
    CircularBuffer cbout(cb_out);

    // Writes chunk c's o slab (rows of this V-block at row stride Vt_full) from the front of cb_out.
    auto write_o = [&](uint32_t c) {
        DeviceZoneScopedN("ow_write");
        const uint32_t row_base = (h * NC + c) * Ct * Vt_full;
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
        for (uint32_t r = 0; r < Ct; r++) {
            const uint32_t dst = row_base + r * Vt_full + vb * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_write(src, o_acc, tb_o, {.offset_bytes = (r * Vt + vt) * tb_o}, {.page_id = dst + vt});
            }
        }
    };

    // o [BH, NC, C, V]: scatter this V-block back — row stride Vt_full, column offset vb*Vt. Chunks 0 .. NC-2 first,
    // each with its own barrier (frees its cb_out slot for compute).
    for (uint32_t c = 0; c + 1 < NC; c++) {
        {
            DeviceZoneScopedN("ow_wait_cb");
            cbout.wait_front(cv);
        }
        write_o(c);
        noc.async_write_barrier();
        cbout.pop_front(cv);
    }

    // final_state [BH, K, V]: same V-block slicing (row stride Vt_full over K rows). The state's rows are pushed during
    // the last chunk's S_new, before that chunk's o, so they go out BEFORE the last o (each K row as soon as it is
    // packed), and the last o and the whole final state share ONE barrier at the end of the op instead of two
    // serialized write + ack round trips (the state used to wait for the last o).
    CircularBuffer cbfs(cb_final);
    {
        DeviceZoneScopedN("ow_fs");
        const uint32_t row_base = h * Kt * Vt_full;
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbfs);
        for (uint32_t r = 0; r < Kt; r++) {
            cbfs.wait_front((r + 1) * Vt);  // rows of the state arrive one K tile-row at a time
            const uint32_t dst = row_base + r * Vt_full + vb * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_write(src, fs_acc, tb_fs, {.offset_bytes = (r * Vt + vt) * tb_fs}, {.page_id = dst + vt});
            }
        }
    }
    if (NC > 0) {
        {
            DeviceZoneScopedN("ow_wait_cb");
            cbout.wait_front(cv);
        }
        write_o(NC - 1);
    }
    noc.async_write_barrier();
    if (NC > 0) {
        cbout.pop_front(cv);
    }
    cbfs.pop_front(kv);
}
