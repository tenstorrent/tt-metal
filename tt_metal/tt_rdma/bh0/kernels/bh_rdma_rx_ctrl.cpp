// SPDX-License-Identifier: Apache-2.0
//
// Phase 3.1e — eth RISC1 as the drainer pool's CONTROL PLANE (off the per-frame data path). In the pool
// architecture the Tensix workers do all per-frame RDMA work (claim -> read -> rkey->MR -> land ->
// complete); this eth kernel does NO per-frame work. It:
//   1. Configures RXQ2 raw BUF_WRAP (the MAC fills the L1 ring) and publishes the produce head PKT_END_CNT
//      (the monotonic frame count the workers bound their claim by) -- a cheap register snapshot.
//   2. Owns the SHARED MR table: on a host/gateway registration doorbell it writes the MR entry into the
//      shared table (its own L1) and bumps the generation counter, so the workers refresh their cache.
//      This is the "RISC1 registers, workers read" authority -- the control plane, not the data plane.
// (ACK / READ_RESP control-op handling reuses the Phase-1 dispatch logic; folded in as a follow-up. The
// point proven here: the eth RISC is the MR/produce-head authority and never touches the WRITE flood.)

#include <cstdint>

#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"
#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_eth_rx.h"

void kernel_main() {
    // arg0 stats base (L1)  arg1 stop flag  arg2 rx_buf byte addr  arg3 rx_buf size bytes
    // arg4 shared MR table L1 addr (this core)  arg5 registration-request L1 addr (this core)
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t rx_buf = get_arg_val<uint32_t>(2);
    const uint32_t rx_buf_size = get_arg_val<uint32_t>(3);
    const uint32_t mr_table = get_arg_val<uint32_t>(4);
    const uint32_t reg_req = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }
    // Registration request: [go, slot, base, len, rkey, access, dest_x, dest_y]. Gen counter = MR slot 63
    // word 0 (workers refresh on a bump). The control plane fulfils a request then clears `go`.
    volatile tt_l1_ptr uint32_t* rr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reg_req);
    volatile tt_l1_ptr uint32_t* gen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + 63u * 32u);

    tt_rdma_rxq_init(TT_RDMA_RX_QUEUE, rx_buf, rx_buf_size, /*wrap=*/1u);  // raw BUF_WRAP; MAC lands freely
    const uint32_t qb = TT_ETH_RXQ0_BASE + TT_RDMA_RX_QUEUE * TT_ETH_RXQ_STRIDE;

    uint32_t iters = 0, n_reg = 0;
    for (;;) {
        // Publish the produce head (+ diagnostics) -- the workers read stats[2] = PKT_END_CNT.
        stats[0] = TT_ETH_REG32(qb + TT_ETH_RXQ_WORD_CNT);
        stats[1] = TT_ETH_REG32(qb + TT_ETH_RXQ_BYTE_CNT);
        stats[2] = TT_ETH_REG32(qb + TT_ETH_RXQ_PKT_END_CNT);  // produce head
        stats[3] = TT_ETH_REG32(qb + TT_ETH_RXQ_PACKET_DROP_CNT);
        stats[4] = TT_ETH_REG32(qb + TT_ETH_RXQ_BUF_PTR);
        stats[8] = ++iters;
        stats[9] = n_reg;  // MR registrations fulfilled by the control plane

        // Control op: MR registration. On the doorbell, write the MR entry into the shared table and bump
        // the generation so every worker refreshes its cache -- RISC1 is the registration authority.
        if (rr[0] != 0u) {
            const uint32_t slot = rr[1];
            if (slot < 62u) {  // slots 62 (reg req) + 63 (gen) are reserved
                volatile tt_l1_ptr uint32_t* mr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + slot * 32u);
                mr[0] = rr[2];         // base
                mr[2] = rr[3];         // length
                mr[4] = rr[4];         // rkey
                mr[5] = rr[5];         // access
                mr[6] = rr[6];         // dest NoC x
                mr[7] = rr[7];         // dest NoC y
                gen[0] = gen[0] + 1u;  // bump generation -> workers refresh
                ++n_reg;
            }
            rr[0] = 0u;  // clear the doorbell
        }

        if (stop != nullptr && *stop != 0) {
            break;
        }
    }
    stats[9] = n_reg;
}
