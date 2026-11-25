// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem x injector, multicast half (row 0, RISCV_1): the send side of `x_round_multicast_block`.
// X_INJECTORS rotating injectors (each its own Mcast family, alternating NoCs): injector `idx` owns the global
// rounds r = idx, idx + NI, ... Every round its DRAM half (q_stem_injector_reader.cpp) pushed into cb_x_round is
// multicast (Mcast helper, Counter data-ready signal) into slot r % x_round_depth of every stage-1 core's
// cb_x_round (the CB is declared on rect + injectors: one base address). Flow control: the receivers credit each
// slot to its injector as soon as it is free (cumulative count on this core's credit semaphore), so the injectors'
// rounds overlap instead of waiting for a per-round handshake.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/ttnn/operations/q_stem/perf_experiments/x_supply/q_stem_v/kernels/q_stem_common.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_rounds = get_compile_time_arg_val(3);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val(Q_STEM_X_FAMILY_ARG("_ct_offset")),
        get_named_compile_time_arg_val(Q_STEM_X_FAMILY_ARG("_rt_offset"))>();
    constexpr uint32_t idx = get_compile_time_arg_val(4);  // this injector
    constexpr uint32_t num_injectors = get_compile_time_arg_val(5);
    constexpr uint32_t depth = get_compile_time_arg_val(6);      // x_round_depth (receivers' slots)
    constexpr uint32_t receivers = get_compile_time_arg_val(7);  // credits per round
    constexpr uint32_t sem_x_credit = get_compile_time_arg_val(8);
    constexpr uint32_t round_bytes = round_tiles * x_tile_bytes;

    constexpr bool x_split = get_named_compile_time_arg_val("x_split") != 0;
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x_credit));
    if constexpr (x_split) {
        // X_SPLIT (x_supply candidate): this injector multicasts tiles [t_lo, t_lo + t_n) of EVERY round (raw linked
        // multicast, <= 16 KiB commands), then publishes its sent-round count on its ready counter (plain multicast
        // write of a local word behind the payload; receivers wait_min, never clear). Bypasses the Mcast helper: its
        // Flag allows one outstanding round per family, its Counter costs a write + atomic barrier per round.
        constexpr uint32_t sem_x_ready = get_named_compile_time_arg_val("sem_x_ready");
        constexpr uint32_t t_lo = get_named_compile_time_arg_val("x_t_lo");
        constexpr uint32_t t_n = get_named_compile_time_arg_val("x_t_n");
        constexpr uint32_t mx0 = get_named_compile_time_arg_val("x_mc_x0");
        constexpr uint32_t my0 = get_named_compile_time_arg_val("x_mc_y0");
        constexpr uint32_t mx1 = get_named_compile_time_arg_val("x_mc_x1");
        constexpr uint32_t my1 = get_named_compile_time_arg_val("x_mc_y1");
        constexpr uint32_t num_dests = get_named_compile_time_arg_val("x_mc_dests");
        constexpr uint32_t lane_off = t_lo * x_tile_bytes;
        constexpr uint32_t lane_bytes = t_n * x_tile_bytes;
        const uint32_t ready = get_semaphore(sem_x_ready + idx);
        const uint64_t ready_mc = get_noc_multicast_addr(mx0, my0, mx1, my1, ready, noc_index);
        uint32_t base = 0;
        MaybeDeviceZoneScope("INJ_X_MCAST");
        for (uint32_t r = 0; r < num_rounds; ++r) {
            cb_wait_front(cb_x_round, round_tiles);
            const uint32_t src0 = get_read_ptr(cb_x_round);
            if (r == 0) {
                base = src0;
            }
            noc_semaphore_wait_min(credit, receivers * (r + 1));
#ifndef ABLATE_X_MCAST
            uint32_t src = src0 + lane_off, dst = base + (r % depth) * round_bytes + lane_off, left = lane_bytes;
            while (left > 0) {
                const uint32_t b = left < NOC_MAX_BURST_SIZE ? left : NOC_MAX_BURST_SIZE;
                while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
                ncrisc_noc_fast_write<noc_mode>(
                    noc_index,
                    write_cmd_buf,
                    src,
                    get_noc_multicast_addr(mx0, my0, mx1, my1, dst, noc_index),
                    b,
                    NOC_MULTICAST_WRITE_VC,
                    true /* mcast */,
                    true /* linked: the ready write below ends the sequence */,
                    num_dests,
                    true /* path reserve */,
                    false /* non-posted */);
                src += b;
                dst += b;
                left -= b;
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready) = r + 1;
            noc_semaphore_set_multicast(ready, ready_mc, num_dests, false);
            noc_async_writes_flushed();  // payload source + ready word reused only after they left
#endif
            cb_pop_front(cb_x_round, round_tiles);
        }
        noc_async_write_barrier();
    } else {
        Noc noc;
        auto pipe = mc.sender(noc);
        uint32_t base = 0;  // receivers' slot 0 (= this core's cb_x_round base: the CB is fresh at the first wait)
        uint32_t k = 0;
        MaybeDeviceZoneScope("INJ_X_MCAST");  // x rounds multicast (waits on the injector reader and receiver credits)
        for (uint32_t r = idx; r < num_rounds; r += num_injectors, ++k) {
            cb_wait_front(cb_x_round, round_tiles);
            const uint32_t src = get_read_ptr(cb_x_round);
            if (k == 0) {
                base = src;
            }
            noc_semaphore_wait_min(credit, receivers * (k + 1));  // every receiver's slot r % depth is free
#ifndef ABLATE_X_MCAST
            pipe.send(src, base + (r % depth) * round_bytes, round_bytes);  // multicast, then data-ready flag
#endif
            cb_pop_front(cb_x_round, round_tiles);
        }
    }
}
