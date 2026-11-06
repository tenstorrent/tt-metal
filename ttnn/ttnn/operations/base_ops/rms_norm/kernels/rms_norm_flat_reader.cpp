// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm FLAT reader (NCRISC) — HEIGHT_SHARDED ROW_MAJOR, few sticks per Tensix core.
//
// A stick of W datums is contiguous in the shard, so it is read by compute as ceil(W / 1024) "flat tiles" (1024
// consecutive datums each, in memory order — the face/row placement inside a tile is irrelevant for an elementwise
// square, a whole-tile sum and an elementwise multiply by a gamma laid out the same way). No tilize.
//   cb_x IS the input shard (page = gcd(stick pitch, tile bytes), so flat tile j
//       of stick s is page s * stick_pages + j * tile_pages). The reader only publishes it. When W % 1024 != 0 the
//       stick's last partial flat tile (W % 1024 datums) is copied (one local NoC read per stick) into a zero-filled
//       one-tile slot of cb_x_rm — the in-place partial tile would read the next stick / past the shard end.
// Raw NoC (no kernel_lib local-copy helper): one tail-piece local read per stick (pitch-strided source, tile-strided
// destination) — dataflow_kernel_lib's local copy fixes the source and steps only the destination.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "rms_norm_common.hpp"

using namespace rms_norm_cb;

constexpr uint32_t n_body = get_compile_time_arg_val(0);        // flat tiles per stick read from cb_x
constexpr bool has_tail_cb = get_compile_time_arg_val(1) != 0;  // W % 1024 != 0: tail tile via cb_x_rm
constexpr uint32_t tail_bytes = get_compile_time_arg_val(2);    // valid bytes of the tail tile
constexpr uint32_t x_pitch = get_compile_time_arg_val(3);       // shard stick pitch (aligned page bytes)
constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);    // flat tile bytes (1024 datums)
constexpr uint32_t stick_pages = get_compile_time_arg_val(5);   // cb_x pages per stick
[[maybe_unused]] constexpr uint32_t stick_bytes = get_compile_time_arg_val(6);  // W * elem

#ifdef RMS_ABLATE_DM
constexpr bool ablate_dm = true;
#else
constexpr bool ablate_dm = false;
#endif

void kernel_main() {
#ifdef RMS_ABLATE_EMPTY
    return;
#endif
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t sticks = get_arg_val<uint32_t>(1);
    Noc noc;
    {
        MaybeDeviceZoneScope("rd_block");
        // zero-copy body: cb_x is backed on the shard — publish the valid sticks' pages
        cb_reserve_back(cb_x, sticks * stick_pages);
        cb_push_back(cb_x, sticks * stick_pages);
        if constexpr (has_tail_cb) {
            CircularBuffer cb(cb_x_rm);
            cb_reserve_back(cb_x_rm, sticks);
            const uint32_t base = get_write_ptr(cb_x_rm);
            noc.async_write_zeros(cb, sticks * tile_bytes);
            noc.write_zeros_l1_barrier();
            if constexpr (!ablate_dm) {
                for (uint32_t s = 0; s < sticks; ++s) {
                    noc_async_read(
                        get_noc_addr(x_addr + s * x_pitch + n_body * tile_bytes), base + s * tile_bytes, tail_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x_rm, sticks);
        }
    }
    // 1.0 scaler of compute's FPU whole-tile sum (reduce_tile SCALAR)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_ones, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_SCALAR>();
}
