// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm FLAT writer (BRISC) — HEIGHT_SHARDED ROW_MAJOR, few sticks per Tensix core.
//
// Flat gamma: the RM gamma is ONE DRAM page of W datums, which IS ceil(W / 1024) flat tiles in the same datum order
//   as a flat x stick — no faces-0/1 re-layout, no expand, no streaming (W * elem bytes resident: 16 KB at fp32
//   W = 4096, vs. one half tile per 32 columns = 16x that for the tilized row-broadcast layout). One DRAM read of
//   the page by the multicast sender (kernel_lib Mcast, one group of every Tensix core — the op's
//   RMS_GAMMA_MCAST k = 1 channel), one multicast of the same bytes; receivers wait. Datums past W in the last flat
//   tile are don't-care (their outputs are never written back).
// Store:
//   cb_out IS the output shard — compute packs the full flat tiles in place; the stick's tail flat tile
//       (W % 1024 != 0) comes through cb_out_rm and its valid bytes are written back (one local NoC write per stick).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#ifdef RMS_GAMMA_MCAST
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#endif
#include "rms_norm_common.hpp"

using namespace rms_norm_cb;

constexpr uint32_t n_body = get_compile_time_arg_val(0);
constexpr bool has_tail_cb = get_compile_time_arg_val(1) != 0;
constexpr uint32_t tail_bytes = get_compile_time_arg_val(2);
constexpr uint32_t out_pitch = get_compile_time_arg_val(3);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
constexpr uint32_t stick_pages = get_compile_time_arg_val(5);
[[maybe_unused]] constexpr uint32_t stick_bytes = get_compile_time_arg_val(6);  // W * elem
constexpr uint32_t g_read_bytes = get_compile_time_arg_val(7);  // gamma aligned page bytes (one read, one multicast)
constexpr uint32_t gamma_tiles = get_compile_time_arg_val(8);   // flat gamma tiles = ceil(W / 1024)
constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(9);
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<10>();

#ifdef RMS_ABLATE_DM
constexpr bool ablate_dm = true;
#else
constexpr bool ablate_dm = false;
#endif

#ifdef RMS_GAMMA_MCAST
constexpr auto gamma_mcast = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("gamma_mcast_ct_offset"),
    get_named_compile_time_arg_val("gamma_mcast_rt_offset")>();
#endif

void kernel_main() {
#ifdef RMS_ABLATE_EMPTY
    return;
#endif
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t sticks = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(2);

#ifdef RMS_HAS_GAMMA
    {
        cb_reserve_back(cb_gamma, gamma_tiles);
        const uint32_t base = get_write_ptr(cb_gamma);
        {
            MaybeDeviceZoneScope("wr_gamma_issue");
#ifdef RMS_GAMMA_MCAST
            const bool reads = gamma_mcast.can_send();
#else
            constexpr bool reads = true;
#endif
            if (reads) {
                const auto g_acc = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
                noc_async_read(g_acc.get_noc_addr(0), base, g_read_bytes);
            }
        }
        MaybeDeviceZoneScope("wr_gamma_finish");  // barrier (+ multicast send / receive wait)
#ifdef RMS_GAMMA_MCAST
        Noc noc;
        if (auto sender = gamma_mcast.optional_sender(noc)) {
            noc_async_read_barrier();
            sender->send(base, base, g_read_bytes);  // zero fan-out: src == dst, no transfer
        } else if (auto receiver = gamma_mcast.optional_receiver(noc)) {
            receiver->receive();
        }
#else
        noc_async_read_barrier();
#endif
        cb_push_back(cb_gamma, gamma_tiles);
    }
#endif

    MaybeDeviceZoneScope("wr_store");
    // zero-copy: the full flat tiles are already in the output shard
    cb_wait_front(cb_out, sticks * stick_pages);
    if constexpr (has_tail_cb) {
        cb_wait_front(cb_out_rm, sticks);
        const uint32_t l1 = get_read_ptr(cb_out_rm);
        if constexpr (!ablate_dm) {
            for (uint32_t s = 0; s < sticks; ++s) {
                noc_async_write(
                    l1 + s * tile_bytes, get_noc_addr(out_addr + s * out_pitch + n_body * tile_bytes), tail_bytes);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out_rm, sticks);
    }
    cb_pop_front(cb_out, sticks * stick_pages);
    noc_async_write_barrier();
}
