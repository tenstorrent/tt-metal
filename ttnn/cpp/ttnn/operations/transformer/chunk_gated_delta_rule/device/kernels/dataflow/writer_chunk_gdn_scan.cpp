// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) writer, value-parallel. This core produced ONE V-block (vb) of head h:
// columns [vb*Vt, vb*Vt+Vt) of the full V dimension. It writes that slice back into the full
// tensors o [BH, NC, C, V] and final_state [BH, K, V] using DRAM row stride Vt_full.
// EMIT: per chunk also h [BH, NC, K, V] (state entering the chunk) and v_new [BH, NC, C, V],
// same V-slicing, drained per chunk as (h, v_new, o): h_0 from the reader's cb_h0, h_{c>=1} from
// the compute's cb_hout (pushed at the end of chunk c-1).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_final = 27, cb_hout = 0, cb_vnout = 1, cb_h0 = 2;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);  // per-core V-block width (tiles)
    constexpr uint32_t has_s0 = get_compile_time_arg_val(3);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);  // full V (tiles) for row stride
    constexpr uint32_t EMIT = get_compile_time_arg_val(5);
    (void)has_s0;

    constexpr auto o_a = TensorAccessorArgs<6>();
    constexpr auto fs_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto h_a = TensorAccessorArgs<fs_a.next_compile_time_args_offset()>();  // null unless EMIT
    constexpr auto vn_a = TensorAccessorArgs<h_a.next_compile_time_args_offset()>();  // null unless EMIT

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_arg_val<uint32_t>(3);
    const uint32_t fs_addr = get_arg_val<uint32_t>(4);
    const uint32_t h_addr = get_arg_val<uint32_t>(5);
    const uint32_t vn_addr = get_arg_val<uint32_t>(6);

    // Mixed precision: o is bf16 (cb_out), the final state is fp32 (cb_final). Each write MUST use
    // its own tile size, else the fp32 state written at the bf16 stride is garbage (and vice versa).
    const uint32_t tb_o = get_tile_size(cb_out);
    const uint32_t tb_fs = get_tile_size(cb_final);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb_o);
    const auto fs_acc = TensorAccessor(fs_a, fs_addr, tb_fs);
    const auto h_acc = TensorAccessor(h_a, h_addr, tb_fs);  // fp32 like the state
    const auto vn_acc = TensorAccessor(vn_a, vn_addr, tb_o);

    constexpr uint32_t cv = Ct * Vt;  // per-core [C, Vt] output slab
    constexpr uint32_t kv = Kt * Vt;  // per-core [K, Vt] final-state slab

    Noc noc;
    CircularBuffer cbout(cb_out);

    // Scatter one [R, Vt] slab of `cb_id` into a [.., R, V] tensor at tile row `row_base`.
    auto scatter_vslice = [&](uint32_t cb_id, const auto& acc, uint32_t tb, uint32_t row_base, uint32_t R) {
        CircularBuffer cb(cb_id);
        cb.wait_front(R * Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t dst = row_base + r * Vt_full + vb * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_write(src, acc, tb, {.offset_bytes = (r * Vt + vt) * tb}, {.page_id = dst + vt});
            }
        }
        noc.async_write_barrier();
        cb.pop_front(R * Vt);
    };

    // o [BH, NC, C, V]: scatter this V-block back — row stride Vt_full, column offset vb*Vt.
    for (uint32_t c = 0; c < NC; c++) {
        if constexpr (EMIT) {
            // h [BH,NC,K,V]: h_0 staged by the reader, h_{c>=1} teed by compute
            scatter_vslice(c == 0 ? cb_h0 : cb_hout, h_acc, tb_fs, (h * NC + c) * Kt * Vt_full, Kt);
            scatter_vslice(cb_vnout, vn_acc, tb_o, (h * NC + c) * Ct * Vt_full, Ct);  // v_new [BH,NC,C,V]
        }
        cbout.wait_front(cv);
        const uint32_t row_base = (h * NC + c) * Ct * Vt_full;
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
        for (uint32_t r = 0; r < Ct; r++) {
            const uint32_t dst = row_base + r * Vt_full + vb * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_write(src, o_acc, tb_o, {.offset_bytes = (r * Vt + vt) * tb_o}, {.page_id = dst + vt});
            }
        }
        noc.async_write_barrier();
        cbout.pop_front(cv);
    }

    // final_state [BH, K, V]: same V-block slicing (row stride Vt_full over K rows).
    CircularBuffer cbfs(cb_final);
    cbfs.wait_front(kv);
    const uint32_t row_base = h * Kt * Vt_full;
    auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbfs);
    for (uint32_t r = 0; r < Kt; r++) {
        const uint32_t dst = row_base + r * Vt_full + vb * Vt;
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc.async_write(src, fs_acc, tb_fs, {.offset_bytes = (r * Vt + vt) * tb_fs}, {.page_id = dst + vt});
        }
    }
    noc.async_write_barrier();
    cbfs.pop_front(kv);
}
