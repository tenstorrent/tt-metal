// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe 2: NoC COMMAND *ISSUE* COST.
//
// Issue N_CMDS transfers of XFER_BYTES each and one barrier. Two independent sweeps come out of
// the same kernel:
//
//   * COUNT sweep at a deliberately tiny XFER_BYTES (32 B — one NoC flit's worth, so the wire time
//     is negligible against the command-issue time): the slope of ns vs N_CMDS is the per-command
//     ISSUE cost. This is the number the whole "is the op bookkeeping-bound?" question turns on.
//   * SIZE sweep at fixed N_CMDS: the slope of ns vs XFER_BYTES is bandwidth; the intercept at
//     size 0 is issue cost again, cross-checking the first sweep from the other axis.
//
// MODE separates three address-generation strategies that cost different amounts BEFORE the NoC is
// even touched, because the real op pays the first one everywhere:
//   0/2 ACCESSOR — `TensorAccessor::get_noc_addr(page)` per command (the op's own pattern: a
//                  bank-index divide/modulo plus a bank-base lookup per transfer).
//   1/3 FIXED    — the noc address is computed ONCE outside the loop and reused. The gap against
//                  ACCESSOR is exactly what the address arithmetic costs per command.
//   4/5 L1REMOTE — `get_noc_addr(vx, vy, addr)` against a peer core's L1 (trivial addressing, no
//                  bank map). Isolates "NoC issue against L1" from "NoC issue against DRAM".
//
// BARRIER_EACH turns the batch (N commands, one barrier — what a well-pipelined reader does) into
// the serialised form (one barrier per command — what a latency-bound reader pays). The gap is the
// cost of NOT batching, which is a real and separable part of the op's per-M-block bill.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#define MODE_DRAM_READ_ACC 0
#define MODE_DRAM_READ_FIXED 1
#define MODE_DRAM_WRITE_ACC 2
#define MODE_DRAM_WRITE_FIXED 3
#define MODE_L1_READ_REMOTE 4
#define MODE_L1_WRITE_REMOTE 5

constexpr uint32_t MODE = get_compile_time_arg_val(0);
constexpr uint32_t BARRIER_EACH = get_compile_time_arg_val(1);
constexpr uint32_t CB_ID = get_compile_time_arg_val(2);
constexpr uint32_t N_PAGES_IN_BUF = get_compile_time_arg_val(3);
constexpr uint32_t TA_BASE = 4;
constexpr auto buf_args = TensorAccessorArgs<TA_BASE>();

constexpr bool IS_WRITE =
    (MODE == MODE_DRAM_WRITE_ACC) || (MODE == MODE_DRAM_WRITE_FIXED) || (MODE == MODE_L1_WRITE_REMOTE);

void kernel_main() {
    const uint32_t buf_addr = get_arg_val<uint32_t>(0);
    const uint32_t buf_page = get_arg_val<uint32_t>(1);
    const uint32_t peer_vx = get_arg_val<uint32_t>(2);
    const uint32_t peer_vy = get_arg_val<uint32_t>(3);
    const uint32_t peer_l1 = get_arg_val<uint32_t>(4);
    // RUNTIME, not compile-time: the op's transfer counts and sizes are runtime values, and one
    // compiled kernel then serves the whole sweep instead of one JIT build per point.
    const uint32_t n_cmds = get_arg_val<uint32_t>(5);
    const uint32_t xfer_bytes = get_arg_val<uint32_t>(6);

    // L1 scratch that every mode lands in / sources from. Reserved once, never pushed: this probe
    // has no consumer, and the point is to keep the CB cycle OUT of the measured loop.
    cb_reserve_back(CB_ID, 1);
    const uint32_t l1 = get_write_ptr(CB_ID);

    const auto acc = TensorAccessor(buf_args, buf_addr, buf_page);

    // Hoisted address for the FIXED / L1REMOTE modes — computed once so the loop body is the bare
    // NoC command and nothing else.
    uint64_t fixed_noc = 0;
    if constexpr (MODE == MODE_DRAM_READ_FIXED || MODE == MODE_DRAM_WRITE_FIXED) {
        fixed_noc = acc.get_noc_addr(0);
    } else if constexpr (MODE == MODE_L1_READ_REMOTE || MODE == MODE_L1_WRITE_REMOTE) {
        // The peer's scratch lives at the SAME L1 offset as ours: every core in the range gets the
        // identical CB declaration, so this core's own write pointer IS the peer's address. (Same
        // trick the real op uses for its reduce-tree landing addresses, and for the same reason —
        // no address negotiation.) `peer_l1` is therefore unused; kept in the arg list so the host
        // side stays uniform across modes.
        (void)peer_l1;
        fixed_noc = get_noc_addr(peer_vx, peer_vy, l1);
    }

    for (uint32_t i = 0; i < n_cmds; ++i) {
        if constexpr (MODE == MODE_DRAM_READ_ACC) {
            noc_async_read(acc.get_noc_addr(i % N_PAGES_IN_BUF), l1, xfer_bytes);
        } else if constexpr (MODE == MODE_DRAM_WRITE_ACC) {
            noc_async_write(l1, acc.get_noc_addr(i % N_PAGES_IN_BUF), xfer_bytes);
        } else if constexpr (MODE == MODE_DRAM_READ_FIXED || MODE == MODE_L1_READ_REMOTE) {
            noc_async_read(fixed_noc, l1, xfer_bytes);
        } else {  // MODE_DRAM_WRITE_FIXED || MODE_L1_WRITE_REMOTE
            noc_async_write(l1, fixed_noc, xfer_bytes);
        }
        if constexpr (BARRIER_EACH) {
            if constexpr (IS_WRITE) {
                noc_async_write_barrier();
            } else {
                noc_async_read_barrier();
            }
        }
    }
    if constexpr (!BARRIER_EACH) {
        if constexpr (IS_WRITE) {
            noc_async_write_barrier();
        } else {
            noc_async_read_barrier();
        }
    }
}
