// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Builds this chip's routing index, then fills the L1 ring the sender
// on this same core drains.
//
// The prologue is the piece with no counterpart in combine. Combine's input is already grouped by origin
// chip, so a chunk is four words out of a control table. Dispatch's input is token order and a token's
// destination is data-dependent, so the destination-grouped runs the protocol needs have to be
// manufactured here: one pass over (token, top-k slot) that replays the production op's per-expert
// allocator exactly, bucketing the survivors by (destination chip, expert).
//
// Replaying it exactly is what makes the pages byte-identical to the production op, including the rule
// that a token past the buffer's capacity is dropped while its counter still advances.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "dispatch_fabric2d_reader_ct_args.hpp"

constexpr dspf2d::ReaderCtArgs ct{};

namespace {

// The reader's L1 working set, carved out of the control region in one fixed order so every chip lays it
// out identically.
struct Control {
    volatile tt_l1_ptr uint32_t* indices;       // seq_len records, each padded to indices_pad_stride
    volatile tt_l1_ptr uint32_t* offsets_tbl;   // extent x num_routed_experts, this chip's row and every other
    volatile tt_l1_ptr uint32_t* counts;        // num_routed_experts
    volatile tt_l1_ptr uint32_t* region;        // num_routed_experts
    volatile tt_l1_ptr int32_t* table;          // num_routed_experts + 1, the trailing sentinel maps to -1
    volatile tt_l1_ptr uint32_t* alloc;         // num_routed_experts, the running per-expert allocator
    volatile tt_l1_ptr uint32_t* chip_experts;  // extent x experts_per_chip, ascending global expert id
    volatile tt_l1_ptr uint32_t* bucket_len;    // extent x experts_per_chip
    volatile tt_l1_ptr uint32_t* entries;       // 2 words per surviving (token, top-k slot)
};

Control carve_control() {
    uint32_t a = ct.control_addr;
    const auto take_bytes = [&](uint32_t bytes) {
        const uint32_t at = a;
        a += bytes;
        return at;
    };
    const auto take = [&](uint32_t words) { return take_bytes(words * 4u); };

    Control c;
    c.indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take_bytes(ct.seq_len * ct.indices_pad_stride));
    c.offsets_tbl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(ct.extent * ct.num_routed_experts));
    c.counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(ct.num_routed_experts));
    c.region = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(ct.num_routed_experts));
    c.table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(take(ct.num_routed_experts + 1));
    c.alloc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(ct.num_routed_experts));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(ct.extent * ct.experts_per_chip));
    c.bucket_len = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(ct.extent * ct.experts_per_chip));
    c.entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(2 * ct.seq_len * ct.topk));
    return c;
}

}  // namespace

void kernel_main() {
    uint32_t rt = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t indices_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t expert_offsets_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t table_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t counts_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t region_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t out_payload_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t out_meta_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t fwd_addr = get_arg_val<uint32_t>(rt++);
    (void)input_addr;
    (void)out_payload_addr;
    (void)out_meta_addr;
    (void)fwd_addr;

    constexpr uint32_t accessor_base = dspf2d::ReaderCtArgs::kCount;
    (void)accessor_base;

    Noc noc;
    Control c = carve_control();

    constexpr auto offsets_args =
        TensorAccessorArgs<dspf2d::ReaderCtArgs::kCount + 0>();  // placeholder: accessor chain wired with the read path
    (void)offsets_args;

    // Placeholder: the routing index, own assignments, relays and the local phase land next. Publishing
    // CMD_END immediately lets the sender open its connection, drain one slot and close, which is what
    // makes the L1 layout, the semaphores, the fabric wiring and both kernels' argument contracts
    // testable before any token moves.
    volatile tt_l1_ptr dspf2d::FwdMetadata* tail =
        reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(ct.ring_addr + ct.token_size_bytes);
    tail->cmd = dspf2d::CMD_END;
    noc_semaphore_inc(get_noc_addr(ct.filled_addr), 1);

    (void)indices_addr;
    (void)expert_offsets_addr;
    (void)table_addr;
    (void)counts_addr;
    (void)region_addr;
    (void)c;

    noc_async_atomic_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr), 0);
}
