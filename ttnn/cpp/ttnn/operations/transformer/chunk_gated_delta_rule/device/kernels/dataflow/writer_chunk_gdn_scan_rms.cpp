// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) writer, value-parallel. This core produced ONE V-block (vb) of head h:
// columns [vb*Vt, vb*Vt+Vt) of the full V dimension. It writes that slice directly into the
// assigned RMS consumer core's bounded L1 staging buffer, and writes final_state to DRAM.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_final = 27;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);  // per-core V-block width (tiles)
    constexpr uint32_t initial_state_mode = get_compile_time_arg_val(3);
    constexpr uint32_t Vt_full = get_compile_time_arg_val(4);  // full V (tiles) for row stride
    constexpr bool state_only = get_compile_time_arg_val(5) == 1;
    static_assert(Ct == 1, "fused scan-to-RMS handoff requires 32-token chunks");
    (void)initial_state_mode;

    constexpr auto fs_a = TensorAccessorArgs<7>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t vb = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t fs_addr = get_arg_val<uint32_t>(3);
    const uint32_t consumer_count = get_arg_val<uint32_t>(4);
    const uint32_t ready_semaphore_id = get_arg_val<uint32_t>(5);

    // The scan output and final state remain FP32; the former is transferred directly to consumer L1.
    const uint32_t tb_o = get_tile_size(cb_out);
    const uint32_t tb_fs = get_tile_size(cb_final);
    const auto fs_acc = TensorAccessor(fs_a, fs_addr, tb_fs);

    constexpr uint32_t cv = Ct * Vt;  // per-core [C, Vt] output slab
    constexpr uint32_t kv = Kt * Vt;  // per-core [K, Vt] final-state slab

    Noc noc;
    CircularBuffer cbout(cb_out);
    const uint32_t staging_l1_base = get_write_ptr(0);

    // Stage this V-block in the assigned consumer's full-V row, then publish readiness.
    for (uint32_t c = 0; c < NC; c++) {
        cbout.wait_front(cv);
        if constexpr (!state_only) {
            const uint32_t wi = h * NC + c;
            const uint32_t consumer = wi % consumer_count;
            const uint32_t local_item = wi / consumer_count;
            const uint32_t noc_x = get_arg_val<uint32_t>(6 + 2 * consumer);
            const uint32_t noc_y = get_arg_val<uint32_t>(7 + 2 * consumer);
            const uint32_t dst_l1 = staging_l1_base + (local_item * Vt_full + vb * Vt) * tb_o;
            const uint64_t dst_noc = get_noc_addr(noc_x, noc_y, dst_l1);
            const uint32_t src = get_read_ptr(cb_out);
            noc_async_write(src, dst_noc, Vt * tb_o);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(noc_x, noc_y, get_semaphore(ready_semaphore_id)), 1);
        }
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
