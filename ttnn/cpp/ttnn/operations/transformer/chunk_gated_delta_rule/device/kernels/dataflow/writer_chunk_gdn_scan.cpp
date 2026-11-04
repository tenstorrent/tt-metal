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
#ifdef GDN_SNAPSHOT
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "chunk_gdn_handoff.hpp"
#endif

constexpr uint32_t cb_out = 16, cb_final = 27;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);       // per-core V-block width (tiles)
    constexpr uint32_t Vt_full = get_compile_time_arg_val(3);  // full V (tiles) for row stride

    constexpr auto o_a = TensorAccessorArgs<4>();
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

#ifdef GDN_SNAPSHOT
    // Mid-scan snapshots (fused program, ChunkGdnParams::num_snapshots = S_SNAP > 0). CT args after the two accessors:
    // S_SNAP, BH, then the snapshot_state and snapshot_index accessors; RT args 5 / 6 their addresses. The S chunk
    // indices are read from page 0 of snapshot_index into the control page, which the scan compute reads
    // (read_tile_value) and pops; this writer keeps its own copy. Slot ks of snapshot_state [S, BH, K, V] receives
    // this core's [K, Vt] slice of the state packed after chunk snap_c[ks] (cb_snap), in the final state's layout.
    constexpr uint32_t S_SNAP = get_compile_time_arg_val(fs_a.next_compile_time_args_offset());
    constexpr uint32_t BH = get_compile_time_arg_val(fs_a.next_compile_time_args_offset() + 1);
    constexpr auto ss_a = TensorAccessorArgs<fs_a.next_compile_time_args_offset() + 2>();
    constexpr auto si_a = TensorAccessorArgs<ss_a.next_compile_time_args_offset()>();
    static_assert(S_SNAP >= 1 && S_SNAP <= gdn_handoff::kMaxSnapshots, "snapshot slots exceed the control page");
    const uint32_t ss_addr = get_arg_val<uint32_t>(5);
    const uint32_t si_addr = get_arg_val<uint32_t>(6);
    const auto ss_acc = TensorAccessor(ss_a, ss_addr, tb_fs);
    const auto si_acc = TensorAccessor(si_a, si_addr);
    uint32_t snap_c[S_SNAP];
    {
        CircularBuffer cbctl(gdn_handoff::kCbSnapCtl);
        cbctl.reserve_back(1);
        const uint32_t ctl_ptr = cbctl.get_write_ptr();
        noc.async_read(si_acc, CoreLocalMem<uint32_t>(ctl_ptr), S_SNAP * sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < S_SNAP; i++) {
            snap_c[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl_ptr)[i];
        }
        cbctl.push_back(1);
    }
    uint32_t ks = 0;
    CircularBuffer cbsnap(gdn_handoff::kCbSnap);
#endif

    // o [BH, NC, C, V]: scatter this V-block back — row stride Vt_full, column offset vb*Vt.
    for (uint32_t c = 0; c < NC; c++) {
        {
            DeviceZoneScopedN("ow_wait_cb");
            cbout.wait_front(cv);
        }
        DeviceZoneScopedN("ow_write");
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
#ifdef GDN_SNAPSHOT
        // The compute pushed chunk c's o before its updated state, so the snapshot of chunk c (if any) follows here.
        if (ks < S_SNAP && snap_c[ks] == c) {
            cbsnap.wait_front(kv);
            const uint32_t snap_base = (ks * BH + h) * Kt * Vt_full;
            auto ssrc = use<CircularBuffer::AddrSelector::READ_PTR>(cbsnap);
            for (uint32_t r = 0; r < Kt; r++) {
                const uint32_t dst = snap_base + r * Vt_full + vb * Vt;
                for (uint32_t vt = 0; vt < Vt; vt++) {
                    noc.async_write(
                        ssrc, ss_acc, tb_fs, {.offset_bytes = (r * Vt + vt) * tb_fs}, {.page_id = dst + vt});
                }
            }
            noc.async_write_barrier();
            cbsnap.pop_front(kv);
            ks++;
        }
#endif
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
