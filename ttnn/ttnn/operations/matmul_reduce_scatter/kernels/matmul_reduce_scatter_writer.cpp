// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — matmul-core writer (BRISC, NoC1): handoff_or_store_block per finished entry, in order.
//
// cb_finished is backed by the staging tensor (same L1 address on every core and chip); entry e of this core sits in
// slot e mod depth_handoff. start / relay: one mailbox increment to the port (direction, link of this core); the slot
// is popped (FIFO) once that port acks (program-local consumed_fwd / consumed_bwd). fin: the valid tiles go to the
// output with one write barrier. Signalling never waits on consumption: a pop is forced only when every slot is
// outstanding.
//
// Raw-API note (design "Helpers considered and rejected"): the handoff is many-to-one with per-producer acks; no
// helper expresses it.

#include <cstdint>
#ifdef MRS_ZONES  // perf instrumentation only
#include <tools/profiler/kernel_profiler.hpp>
#define MRS_ZONE(name) DeviceZoneScopedN(name)
#else
#define MRS_ZONE(name)
#endif
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_finished = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t core_n = get_compile_time_arg_val(2);
    constexpr uint32_t out_n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t depth_handoff = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_entries = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr uint32_t KIND_FWD = 0, KIND_BWD = 1, KIND_FIN = 2;
    constexpr uint32_t consumed_fwd_id = 0, consumed_bwd_id = 1;

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);
    const uint32_t m_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t n_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t fwd_port_xy = get_arg_val<uint32_t>(arg++);
    const uint32_t bwd_port_xy = get_arg_val<uint32_t>(arg++);
    const uint32_t mailbox_slot_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t kinds_idx = arg;

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    volatile tt_l1_ptr uint32_t* consumed_fwd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(consumed_fwd_id));
    volatile tt_l1_ptr uint32_t* consumed_bwd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(consumed_bwd_id));
    const uint64_t fwd_mailbox = get_noc_addr(fwd_port_xy & 0xFFFF, fwd_port_xy >> 16, mailbox_slot_addr);
    const uint64_t bwd_mailbox = get_noc_addr(bwd_port_xy & 0xFFFF, bwd_port_xy >> 16, mailbox_slot_addr);
    const uint32_t finished_base = get_read_ptr(cb_finished);  // slot 0 (read pointer is at the base at launch)

    uint32_t queue[depth_handoff];
    uint32_t head = 0, outstanding = 0, acked_fwd = 0, acked_bwd = 0;
    auto pop_oldest = [&]() {
        MRS_ZONE("ACK_WAIT");
        const uint32_t kind = queue[head];
        head = head + 1 == depth_handoff ? 0 : head + 1;
        if (kind == KIND_FWD) {
            noc_semaphore_wait_min(consumed_fwd, ++acked_fwd);
        } else if (kind == KIND_BWD) {
            noc_semaphore_wait_min(consumed_bwd, ++acked_bwd);
        }
        cb_pop_front(cb_finished, P);
        --outstanding;
    };

    for (uint32_t e = 0; e < num_entries; ++e) {
        const uint32_t kind = get_arg_val<uint32_t>(kinds_idx + e);
        const uint32_t slot = finished_base + (e % depth_handoff) * P * tile_bytes;
        if (kind == KIND_FIN) {
            // stream the output: write each valid tile as soon as the finish has packed it, so the DRAM writes of
            // every core overlap the finish adds instead of all landing after them (the call's tail)
            for (uint32_t i = 0; i < m_valid; ++i) {
                const uint32_t page = (row0 + i) * out_n_tiles + col0;
                {
                    MRS_ZONE("FIN_WAIT");
                    cb_wait_front(cb_finished, outstanding * P + i * core_n + n_valid);
                }
                for (uint32_t j = 0; j < n_valid; ++j) {
                    noc_async_write(slot + (i * core_n + j) * tile_bytes, out.get_noc_addr(page + j), tile_bytes);
                }
            }
            cb_wait_front(cb_finished, (outstanding + 1) * P);
            noc_async_write_barrier();
        } else {
            {
                MRS_ZONE("FIN_WAIT");
                cb_wait_front(cb_finished, (outstanding + 1) * P);
            }
            noc_semaphore_inc(kind == KIND_FWD ? fwd_mailbox : bwd_mailbox, 1);
        }
        queue[(head + outstanding) % depth_handoff] = kind;
        ++outstanding;
        if (outstanding == depth_handoff) {
            pop_oldest();
        }
    }
    while (outstanding > 0) {
        pop_oldest();
    }
    noc_async_atomic_barrier();
}
